from youtube_comment_downloader import YoutubeCommentDownloader
url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
print('start')
d = YoutubeCommentDownloader()
it = d.get_comments_from_url(url)
import itertools
print(list(itertools.islice(it, 3)))
print('end')
