import datetime
import json
import os
from googleapiclient.discovery import build
import webbrowser

API_KEY = 'ENTER YOUR API KEY HERE'
FETCH_DAYS = 30

youtube = build('youtube', 'v3', developerKey=API_KEY)


def get_video_info(video_id):
    request = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    )

    response = request.execute()['items'][0]

    return {
        "video_id": response['id'],
        "published_at": response['snippet']['publishedAt'],
        "channel_id": response['snippet']['channelId'],
        "title": response['snippet']['title'],
        "description": response['snippet']['description'],
        "tags": response['snippet']['tags'] if 'tags' in response['snippet'] else None,
        "category_id": response['snippet']['categoryId'],
        "default_language": response['snippet']['defaultLanguage'] if 'defaultLanguage' in response['snippet'] else None,
        "comment_count": response['statistics']['commentCount'],
        "like_count": response['statistics']['likeCount'],
        "view_count": response['statistics']['viewCount'],
    }


def get_video_comments(api_service, path, fetch_days, count, **kwargs):
    results = api_service.list(**kwargs).execute()

    while results:
        print(".", end="", flush=True)
        data = None
        for item in results['items']:
            with open(f"{path}/comment_{count}.json", "w", encoding="utf-8") as f:
                data = {
                    "comment_id": item['id'],
                    "text": item['snippet']['topLevelComment']['snippet']['textOriginal'],
                    "like_count": item['snippet']['topLevelComment']['snippet']['likeCount'],
                    "published_at": item['snippet']['topLevelComment']['snippet']['publishedAt'],
                    "updated_at": item['snippet']['topLevelComment']['snippet']['updatedAt'],
                }
                json.dump(data, f, indent=2, ensure_ascii=False)
            count += 1

        if fetch_days is not None and data is not None and (datetime.datetime.now() - datetime.datetime.strptime(data['updated_at'], "%Y-%m-%dT%H:%M:%SZ")).days > fetch_days:
            break

        if 'nextPageToken' in results:
            kwargs['pageToken'] = results['nextPageToken']
            results = api_service.list(**kwargs).execute()
        else:
            break


def get_channel_video_ids(channel_id, page_token=None):

    request = youtube.search().list(
        part="id",
        channelId=channel_id,
        maxResults=50,
        order="date",
        pageToken=page_token,
    )

    response = request.execute()

    if 'items' not in response:
        return []

    ids = [item['id']['videoId']for item in response['items'] if "videoId" in item['id']]

    if 'nextPageToken' in response:
        return ids + get_channel_video_ids(channel_id, response['nextPageToken'])

    return ids


if __name__ == '__main__':
    VIDEO_ID = 'zuzWawCDEvo'
    result = get_video_info(VIDEO_ID)
    CHANNEL_ID = result['channel_id']

    video_ids = get_channel_video_ids(CHANNEL_ID)
    video_ids = [
        "IV16BW5eJwQ",
        "G2HhXQPLrbo",
        "FkbIDH26HNo",
        "CR3JVvHBzPU",
        "BZRjkdXYPMM",
        "BvvghiRwvjI"
    ]

    for i, vid in enumerate(video_ids):
        print(f"processing video {i + 1}/{len(video_ids)}")
        try:
            result = get_video_info(vid)
            if not os.path.exists(f"data/cover/{vid}"):
                os.makedirs(f"data/cover/{vid}")
            with open(f"data/cover/{vid}/overview.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            get_video_comments(youtube.commentThreads(),
                               path=f"data/cover/{vid}",
                               fetch_days=None,
                               count=0,
                               part='snippet',
                               videoId=vid,
                               textFormat='plainText',
                               order='time')
        except Exception as e:
            print(e)
            continue

    # for i, vid in enumerate(os.listdir("data/cover")):
    #     url = f"https://www.youtube.com/watch?v={vid}"
    #     print(vid)
    #     with open(f"data/cover/{vid}/overview.json", "r", encoding="utf-8") as f:
    #         data = json.load(f)
    #         if 'original_video_id' in data:
    #             continue
    #
    #     webbrowser.open_new_tab(url)
    #
    #     if vid == "9cTtWgTZDFM":
    #         data['original_video_id'] = "wEWF2xh5E8s"
    #         with open(f"data/cover/{vid}/overview.json", "w", encoding="utf-8") as f:
    #             json.dump(data, f, indent=2, ensure_ascii=False)
    #         continue
    #
    #     if vid == "2I64kEH8jso":
    #         VIDEO_ID = "Dgvf6hlvUI0"
    #     else:
    #         VIDEO_ID = input(f"Enter original video id by cover {vid}: ")
    #
    #     while True:
    #         try:
    #             result = get_video_info(VIDEO_ID)
    #             data['original_video_id'] = "tt2k8PGm-TI"
    #
    #             if not os.path.exists(f"data/original/{VIDEO_ID}"):
    #                 os.makedirs(f"data/original/{VIDEO_ID}")
    #             with open(f"data/original/{VIDEO_ID}/overview.json", "w", encoding="utf-8") as f:
    #                 json.dump(result, f, indent=2, ensure_ascii=False)
    #
    #             get_video_comments(youtube.commentThreads(),
    #                                path=f"data/original/{VIDEO_ID}",
    #                                fetch_days=None,
    #                                count=0,
    #                                part='snippet',
    #                                videoId=VIDEO_ID,
    #                                textFormat='plainText',
    #                                order='time')
    #
    #             with open(f"data/cover/{vid}/overview.json", "w", encoding="utf-8") as f:
    #                 json.dump(data, f, indent=2, ensure_ascii=False)
    #             break
    #         except Exception as e:
    #             print(e)
    #             VIDEO_ID = input(f"Enter original video id by cover {vid}: ")
    #             continue

