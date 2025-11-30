import gdown

url = 'https://drive.google.com/uc?id=1rJO1QSm1Rqp6JmHHNKcPni6maqbgKj0R'
output = 'src/lib/asyrp/pretrained/celebahq_pt2.pt'
gdown.download(url, output, quiet=False)

