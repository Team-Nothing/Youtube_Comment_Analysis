import json
import os
import webbrowser

if __name__ == "__main__":
    for video in os.listdir("data/cover"):
        with open(f"data/cover/{video}/overview.json", encoding="utf-8") as f:
            data = json.load(f)

        print(video, data["comment_count"], len(os.listdir(f"data/cover/{video}")))
