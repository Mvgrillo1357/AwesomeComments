import re
import tweepy

client = tweepy.Client(
    bearer_token="AAAAAAAAAAAAAAAAAAAAABCcbgEAAAAARQajLQw1dIeaFkDX9Yry318YX1Y%3DYgvwYZW2q6VdcOdCOKkHrVM2D6UTR6nKt4EyX8DiiKcy3FYBZX")

def get_tweet_conversation(tweet_url):
    tweet_id = re.search('/status/(\d+)', tweet_url).group(1)
    tweet_response = client.get_tweet(id=tweet_id, tweet_fields=["conversation_id"])
    conversation = []
    conversation.append({
        "id": tweet_response.data.id,
        "text": tweet_response.data.text
    })

    conversation_id = tweet_response.data.conversation_id
    conversation_response = client.search_recent_tweets(
        query=f"conversation_id:{conversation_id} lang:en", 
        sort_order="relevancy",
        max_results=14,
        expansions=["author_id"], 
        user_fields=["name"])

    print(conversation_response)

    for tweet in conversation_response.data:
        conversation.append({
            "id": tweet.id,
            "text": tweet.text
        })
    return conversation
