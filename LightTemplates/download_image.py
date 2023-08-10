import os
import sys
import argparse
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--save_name', type=str)
    args = parser.parse_args()

    url = args.url

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    payload = {}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7'
    }

    print(f'Downloading image {url}')

    response = requests.request("GET", url, headers=headers, data=payload)

    if response.status_code == 200:
        filename = os.path.join(args.save_path, args.save_name)
        with open(filename, 'wb') as file:
            print(f'Save to {filename}')
            file.write(response.content)
    else:
        print(f'Download failed: {response.status_code} {response.text}')
        sys.exit(1)


if __name__ == "__main__":
    main()
