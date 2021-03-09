import json

import praw
import csv
import pandas
import threading
import numpy

from multiprocessing import Pool
from functools import partial
from itertools import chain

submissions = pandas.read_csv(
    '/mnt/ssd_large/Reddit/submissions/submissions_wallstreetbets.csv',
    names=['type', 'depth', 'parent_id', 'submission_id', 'content', 'author_id', 'timestamp', 'up_votes','ratio'],
    dtype={
        'type': 'string',
        'depth': int,
        'parent_id': 'string',
        'submission_id': 'string',
        'content': 'string',
        'author_id': 'string',
        'timestamp': int,
        'up_votes': int,
        'ratio': float
    }
)

THREADS = 4
CSV_WRITE_LOCK = threading.Lock()

with open('/mnt/ssd_large/Reddit/submissions/users_wallstreetbets.csv', 'a') as csv_file:
    csv_writer = csv.writer(csv_file)

    def write_to_csv(author_rows):
        with CSV_WRITE_LOCK:
            csv_writer.writerows(author_rows)
            csv_file.flush()
            print(f'Wrote {len(author_rows)} authors.')
            
    def fetch_authors(is_fullname, track_authors, author_ids):
        with open('../reddit_creds.json') as creds_file:
            creds = json.load(creds_file)

        reddit = praw.Reddit(
            client_id=creds['client_id'],
            client_secret=creds['client_secret'],
            username=creds['username'],
            password=creds['password'],
            user_agent=creds['user_agent']
        )

        author_rows = []
        seen_authors = set()
        
        for author_id in author_ids:
            try:
                if is_fullname:
                    author = reddit.redditor(fullname='t2_' + author_id)
                else:
                    author = reddit.redditor(author_id)
                    
                author_rows.append([
                    author.id,
                    author.name,
                    author.comment_karma,
                    author.link_karma,
                    author.is_mod,
                    author.is_employee
                ])
                
                if track_authors:
                    seen_authors.add(author.id)
            except:
                print('Error with: ' + author_id)
                
            if author_rows and len(author_rows) % 20 == 0:
                write_to_csv(author_rows)
                author_rows = []
                
        if author_rows:
            write_to_csv(author_rows)
            
        return seen_authors

    # was stupid, extracted user names for submissions, and user ids for comments, so need to do it in 2 stages
    submission_author_names = submissions[submissions.type != 'comment'].author_id.unique()
    print('Total submission authors: ' + str(len(submission_author_names)))

    chunked_author_ids = numpy.array_split(submission_author_names, THREADS)
    with Pool(THREADS) as pool:
        seen_author_ids = set(chain.from_iterable(pool.map(partial(fetch_authors, False, True), chunked_author_ids)))
    
    # fetch comment authors by ids we haven't seen
    comment_author_ids = list(set(submissions[submissions.type == 'comment'].author_id.unique()).difference(seen_author_ids))
    print('Total comment authors: ' + str(len(comment_author_ids)))
    
    chunked_author_ids = numpy.array_split(comment_author_ids, THREADS)
    with Pool(THREADS) as pool:
        pool.map(partial(fetch_authors, True, False), chunked_author_ids)
