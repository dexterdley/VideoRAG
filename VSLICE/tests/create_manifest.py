import glob
import json
import os

output_dir = "./downloads/rival_vids"

downloaded_files = glob.glob(os.path.join(output_dir, "*.info.json"))
downloaded_manifests = []
count = 0
for info_json_path in downloaded_files:
	video_id = info_json_path.split("/")[-1].split(".")[0]
	print('Processing:', video_id)
	if os.path.exists(info_json_path):
		with open(info_json_path, 'r', encoding='utf-8') as f:
			data = json.load(f)
			heatmap = data.get('heatmap')
			
			if heatmap:
				heatmap_filename = os.path.join(output_dir, f"{video_id}_heatmap.json")

			# Build the dictionary to return to the main loop
			manifest_data  = {
				"video_id": video_id,
				"title": data.get('title', 'Unknown'),
				"duration": data.get('duration', 0),
				"video_path": os.path.join(output_dir, f"{video_id}.mp4"),
				"heatmap_path": heatmap_filename,
				"heatmap_points": len(heatmap),
			}
			downloaded_manifests.append(manifest_data)
			count += 1


if os.path.exists("./downloads/rival_vids/manifest_json"):
	print("Old JSON found, removing it")
	os.remove("./downloads/rival_vids/manifest_json")

manifest_path = os.path.join(output_dir, "manifest.json")
with open(manifest_path, "w", encoding='utf-8') as f:
	json.dump(downloaded_manifests, f, indent=4, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"🏁 RESULTS: Processed downloaded {len(downloaded_manifests)} videos.")
print(f"📋 New manifest saved to: {manifest_path}")
print(f"{'='*60}")