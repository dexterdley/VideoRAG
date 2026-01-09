import os
import sys
import json
import shutil
import asyncio
import multiprocessing
import torch # Added for local model
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, cast
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoModelForVision2Seq
import tiktoken
from sentence_transformers import SentenceTransformer # Added for local embeddings

import torchvision.transforms.functional as F
sys.modules['torchvision.transforms.functional_tensor'] = F

# Keep existing relative imports
from ._llm import (
    LLMConfig,
    openai_config,
    # azure_openai_config, # Not needed
    # ollama_config        # Not needed
)
from ._op import (
    chunking_by_video_segments,
    extract_entities,
    get_chunks,
    videorag_query,
    videorag_query_multiple_choice,
)
from ._storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NanoVectorDBVideoSegmentStorage,
    NetworkXStorage,
)
from ._utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    wrap_embedding_func_with_attrs,
    convert_response_to_json,
    always_get_an_event_loop,
    logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)
from ._videoutil import(
    split_video,
    speech_to_text,
    segment_caption,
    merge_segment_information,
    saving_video_segments,
)

@dataclass
class QwenVideoRAG:
    working_dir: str = field(
        default_factory=lambda: f"./videorag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    
    # --- Local Model Configuration (New) ---
    local_llm_path: str = "Qwen/Qwen3-VL-8B-Instruct" # Use Qwen-3 path if available
    local_embedding_path: str = "BAAI/bge-m3"        # Standard local embedding model
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # ---------------------------------------

    # video
    threads_for_split: int = 10
    video_segment_length: int = 30 # seconds
    rough_num_frames_per_segment: int = 5 # frames
    fine_num_frames_per_segment: int = 15 # frames
    video_output_format: str = "mp4"
    audio_output_format: str = "mp3"
    video_embedding_batch_num: int = 2
    segment_retrieval_top_k: int = 4
    video_embedding_dim: int = 1024 # Ensure this matches your local_embedding_path dimension (bge-m3 is 1024)
    
    # query
    retrieval_topk_chunks: int = 2
    query_better_than_threshold: float = 0.2
    
    # graph mode
    enable_local: bool = True
    enable_naive_rag: bool = True

    # text chunking
    chunk_func: Callable = chunking_by_video_segments
    chunk_token_size: int = 1200
    tiktoken_model_name: str = "gpt-4o" 

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # Removed API LLMConfig default, we will override in post_init
    llm: LLMConfig = field(default_factory=lambda: LLMConfig(
        domain="local", 
        model_name="qwen-local", 
        embedding_dim=1024 # Match bge-m3
    ))
    
    entity_extraction_func: callable = extract_entities
    
    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vs_vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBVideoSegmentStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    enable_llm_cache: bool = True

    # extension
    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    def load_caption_model(self, debug=False):
        # caption model (MiniCPM-V)
        if not debug:
            print("Loading Caption Model (MiniCPM-V)...")
            self.caption_model = AutoModel.from_pretrained('.checkpoints/MiniCPM-V-2_6-int4', trust_remote_code=True, dtype=torch.bfloat16)
            self.caption_tokenizer = AutoTokenizer.from_pretrained('.checkpoints/MiniCPM-V-2_6-int4', trust_remote_code=True)
            self.caption_model.to(self.device).eval()
        else:
            self.caption_model = None
            self.caption_tokenizer = None

    def load_local_llm(self):
        """Loads Qwen and Embedding models locally"""
        print(f"Loading Local LLM: {self.local_llm_path}...")
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(self.local_llm_path, trust_remote_code=True)
        self.qwen_model = AutoModelForVision2Seq.from_pretrained(
            self.local_llm_path, 
            device_map="auto", 
            dtype=torch.bfloat16,
            trust_remote_code=True
        ).eval()

        print(f"Loading Local Embeddings: {self.local_embedding_path}...")
        self.embed_model = SentenceTransformer(self.local_embedding_path, device=self.device)

    async def _local_qwen_chat(self, prompt: str, system_prompt: str = None, history: list = None, **kwargs):
        """Async wrapper for Qwen Inference"""
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        text = self.qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.qwen_tokenizer([text], return_tensors="pt").to(self.device)

        # Run inference in a thread to avoid blocking asyncio loop
        def run_inference():
            with torch.no_grad():
                generated_ids = self.qwen_model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=2048,
                    temperature=0.7
                )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            return self.qwen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return await asyncio.to_thread(run_inference)

    async def _local_embedding_func(self, texts: list[str]):
        """Async wrapper for Local Embeddings"""
        def run_embed():
            # Normalize embeddings is important for cosine similarity
            embeddings = self.embed_model.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()
        
        return await asyncio.to_thread(run_embed)

    def __post_init__(self):
        self.load_local_llm()
        
        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            os.makedirs(self.working_dir)

        self.video_path_db = self.key_string_value_json_storage_cls(
            namespace="video_path", global_config=asdict(self)
        )
        self.video_segments = self.key_string_value_json_storage_cls(
            namespace="video_segments", global_config=asdict(self)
        )
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )
        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            ) if self.enable_llm_cache else None
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )

        # --- FIX IS HERE: Added **kwargs to lambda ---
        # The library tries to pass 'model_name' when calling this, so we must accept it.
        local_embed_wrapper = lambda texts, **kwargs: self._local_embedding_func(texts)
        
        self.embedding_func = limit_async_func_call(10)(wrap_embedding_func_with_attrs(
                embedding_dim = self.video_embedding_dim,
                max_token_size = 8192,
                model_name = self.local_embedding_path)(local_embed_wrapper))

        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            ) if self.enable_local else None
        )
        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            ) if self.enable_naive_rag else None
        )
        self.video_segment_feature_vdb = (
            self.vs_vector_db_storage_cls(
                namespace="video_segment_feature",
                global_config=asdict(self),
                embedding_func=None,
            )
        )
        
        async def qwen_wrapper(prompt, system_prompt=None, history=None, **kwargs):
            return await self._local_qwen_chat(prompt, system_prompt, history)

        self.llm.best_model_func = limit_async_func_call(1)(
            partial(qwen_wrapper, hashing_kv=self.llm_response_cache)
        )
        self.llm.cheap_model_func = limit_async_func_call(1)(
            partial(qwen_wrapper, hashing_kv=self.llm_response_cache)
        )
        # -----------------------------------------------

    def insert_video(self, video_path_list=None):
        loop = always_get_an_event_loop()
        for video_path in video_path_list:
            # Step0: check the existence
            video_name = os.path.basename(video_path).split('.')[0]
            if video_name in self.video_segments._data:
                logger.info(f"Find the video named {os.path.basename(video_path)} in storage and skip it.")
                continue
            loop.run_until_complete(self.video_path_db.upsert(
                {video_name: video_path}
            ))
            
            # Step1: split the videos
            segment_index2name, segment_times_info = split_video(
                video_path, 
                self.working_dir, 
                self.video_segment_length,
                self.rough_num_frames_per_segment,
                self.audio_output_format,
            )
            
            # Step2: obtain transcript with whisper
            transcripts = speech_to_text(
                video_name, 
                self.working_dir, 
                segment_index2name,
                self.audio_output_format
            )
            
            # Step3: saving video segments **as well as** obtain caption with vision language model
            manager = multiprocessing.Manager()
            captions = manager.dict()
            error_queue = manager.Queue()
            
            process_saving_video_segments = multiprocessing.Process(
                target=saving_video_segments,
                args=(
                    video_name,
                    video_path,
                    self.working_dir,
                    segment_index2name,
                    segment_times_info,
                    error_queue,
                    self.video_output_format,
                )
            )
            
            process_segment_caption = multiprocessing.Process(
                target=segment_caption,
                args=(
                    video_name,
                    video_path,
                    segment_index2name,
                    transcripts,
                    segment_times_info,
                    captions,
                    error_queue,
                )
            )
            
            process_saving_video_segments.start()
            process_segment_caption.start()
            process_saving_video_segments.join()
            process_segment_caption.join()
            
            # if raise error in this two, stop the processing
            while not error_queue.empty():
                error_message = error_queue.get()
                with open('error_log_videorag.txt', 'a', encoding='utf-8') as log_file:
                    log_file.write(f"Video Name:{video_name} Error processing:\n{error_message}\n\n")
                raise RuntimeError(error_message)
            
            # Step4: insert video segments information
            segments_information = merge_segment_information(
                segment_index2name,
                segment_times_info,
                transcripts,
                captions,
            )
            manager.shutdown()
            loop.run_until_complete(self.video_segments.upsert(
                {video_name: segments_information}
            ))
            
            # Step5: encode video segment features
            loop.run_until_complete(self.video_segment_feature_vdb.upsert(
                video_name,
                segment_index2name,
                self.video_output_format,
            ))
            
            # Step6: delete the cache file
            video_segment_cache_path = os.path.join(self.working_dir, '_cache', video_name)
            if os.path.exists(video_segment_cache_path):
                shutil.rmtree(video_segment_cache_path)
            
            # Step 7: saving current video information
            loop.run_until_complete(self._save_video_segments())
        
        loop.run_until_complete(self.ainsert(self.video_segments._data))

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        if param.mode == "videorag":
            response = await videorag_query(
                query,
                self.entities_vdb,
                self.text_chunks,
                self.chunks_vdb,
                self.video_path_db,
                self.video_segments,
                self.video_segment_feature_vdb,
                self.chunk_entity_relation_graph,
                self.caption_model, 
                self.caption_tokenizer,
                param,
                asdict(self),
            )
        # NOTE: update here
        elif param.mode == "videorag_multiple_choice":
            response = await videorag_query_multiple_choice(
                query,
                self.entities_vdb,
                self.text_chunks,
                self.chunks_vdb,
                self.video_path_db,
                self.video_segments,
                self.video_segment_feature_vdb,
                self.chunk_entity_relation_graph,
                self.caption_model, 
                self.caption_tokenizer,
                param,
                asdict(self),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    async def ainsert(self, new_video_segment):
        await self._insert_start()
        try:
            # ---------- chunking
            inserting_chunks = get_chunks(
                new_videos=new_video_segment,
                chunk_func=self.chunk_func,
                max_token_size=self.chunk_token_size,
            )
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning(f"All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
            if self.enable_naive_rag:
                logger.info("Insert chunks for naive RAG")
                await self.chunks_vdb.upsert(inserting_chunks)

            # TODO: no incremental update for communities now, so just drop all
            # await self.community_reports.drop()

            # ---------- extract/summary entity and upsert to graph
            logger.info("[Entity Extraction]...")
            maybe_new_kg, _, _ = await self.entity_extraction_func(
                inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                global_config=asdict(self),
            )
            if maybe_new_kg is None:
                logger.warning("No new entities found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg
            # ---------- commit upsertings and indexing
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            await self._insert_done()

    async def _insert_start(self):
        tasks = []
        for storage_inst in [
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        await asyncio.gather(*tasks)

    async def _save_video_segments(self):
        tasks = []
        for storage_inst in [
            self.video_segment_feature_vdb,
            self.video_segments,
            self.video_path_db,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
    
    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
            self.video_segment_feature_vdb,
            self.video_segments,
            self.video_path_db,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)