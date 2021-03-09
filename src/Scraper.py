import re
import csv
import sys
import traceback
import time
import argparse
import json
import threading
import praw
import requests

from collections import deque
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED


INPUT_DATE_FORMAT = '%Y-%m-%d'
CSV_WRITE_LOCK = threading.Lock()


def reddit_client(creds_loc):
    with open(creds_loc) as creds_file:
        creds = json.load(creds_file)

    return praw.Reddit(
        client_id=creds['client_id'],
        client_secret=creds['client_secret'],
        username=creds['username'],
        password=creds['password'],
        user_agent=creds['user_agent']
    )


def write_to_csv(file, writer, rows):
    with CSV_WRITE_LOCK:
        writer.writerows(rows)
        file.flush()


def cleanse_body(body):
    body = body.replace('\n', ' ')
    body = re.sub(' +', ' ', body)

    return body


def submission_to_rows(client, submission_meta):
    submission = client.submission(submission_meta['id'])

    def _author(s):
        try:
            return s.author.name
        except:
            return '<deleted>'

    submission_common = [_author(submission), submission.created_utc, submission.score, submission.upvote_ratio]
    rows = [
        ['submission_title', 0, '', submission.id, cleanse_body(submission.title)] + submission_common,
        ['submission_body', 0, '', submission.id, cleanse_body(submission.selftext)] + submission_common
    ]

    def add_replies(parent, replies, depth):
        for reply in replies:
            try:
                rows.append([
                    'comment',
                    depth,
                    parent.id,
                    reply.id,
                    cleanse_body(reply.body),
                    _author(reply),
                    reply.created_utc,
                    reply.ups,
                    0
                ])

                add_replies(reply, reply.replies, depth + 1)

            except:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print(f"Unexpected error processing replies:")
                traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)

    submission.comments.replace_more(limit=0)
    add_replies(submission, submission.comments, 1)

    print(f'Processed {len(rows)} rows from submission {submission_meta["id"]}.')

    return rows


def query_submissions(subreddit: str, after: datetime, before: datetime, submissions_limit: int):
    after_q = '{:.0f}'.format(after.timestamp())
    before_q = '{:.0f}'.format(before.timestamp() - 1)
    url = f'https://api.pushshift.io/reddit/search/submission/?subreddit={subreddit}&sort=desc&sort_type=num_comments&after={after_q}&before={before_q}&size={submissions_limit}'

    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()['data']
        print(f'Retrieved {len(data)} submissions from {subreddit} between {after} to {before}.')
        return data
    else:
        print(f'Error {resp.status_code} when retrieving submissions from {subreddit} between {after} to {before}.')
        return []


def scrape_submission(creds_loc, window, subreddit: str, submissions_limit: int):
    client = reddit_client(creds_loc)
    submissions = query_submissions(subreddit, window[0], window[1], submissions_limit)
    for submission in submissions:
        try:
            rows = submission_to_rows(client, submission)
            write_to_csv(csv_file, csv_writer, rows)
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(f'Unexpected error processing submission:')
            traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)


def query_windows(
        creds_loc: str,
        earliest: datetime,
        latest: datetime,
        step: timedelta,
        subreddit: str,
        submissions_limit: int,
        max_per_minute=120,
        concurrency=6
):
    executor = ThreadPoolExecutor(concurrency)
    futures = []

    throttle_limit = timedelta(minutes=1)
    last_windows = deque(maxlen=max_per_minute)

    current = latest
    while current >= earliest:
        # see if we hit the max per minute throttle
        if len(last_windows) == max_per_minute:
            elapsed = datetime.now() - last_windows[0]
            if elapsed < throttle_limit:
                # sleep a little if we hit the throttle
                sleep_duration = (throttle_limit - elapsed).seconds + 1
                print(f'Throttling, sleeping for {sleep_duration} seconds.')
                time.sleep(sleep_duration)

        # track the current time for throttling
        last_windows.append(datetime.now())

        # process the next window
        next_start = current - step
        print(f"Processing window {next_start} to {current}.")
        future = executor.submit(scrape_submission, creds_loc, (next_start, current), subreddit, submissions_limit)
        futures.append(future)

        # purge all the results that have finished
        futures = list(filter(lambda f: not f.done(), futures))
        # don't get ahead of ourselves, if we hit the concurrent limit, block for one of them to finish
        if len(futures) >= concurrency:
            print(f'Blocking on future with {len(futures)} jobs queued.')
            wait(futures, return_when=FIRST_COMPLETED)

        current = next_start

    # everything queued up, wait for everything to finish
    wait(futures, return_when=ALL_COMPLETED)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract subreddit submissions to csv.')
    parser.add_argument('subreddit', help='Subreddit to extract.')
    parser.add_argument('--creds_loc', help='Json with reddit API credentials.')
    parser.add_argument('--file_loc', help='Destination csv.')
    parser.add_argument('--start', help=f'Start date as {INPUT_DATE_FORMAT}.')
    parser.add_argument('--end', help=f'End date as {INPUT_DATE_FORMAT}.')
    parser.add_argument('--submissions_limit', type=int, help=f'Max submissions per day.')

    args = parser.parse_args()

    with open(args.file_loc, 'a') as csv_file:
        print(f'Writing submissions to {args.file_loc}.')
        csv_writer = csv.writer(csv_file)

        query_windows(
            args.creds_loc,
            datetime.strptime(args.start, INPUT_DATE_FORMAT),
            datetime.strptime(args.end, INPUT_DATE_FORMAT),
            timedelta(days=1),
            args.subreddit,
            args.submissions_limit
        )
