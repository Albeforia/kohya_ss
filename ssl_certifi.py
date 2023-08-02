import os
import os.path
import ssl
import subprocess
import sys


def main():
    openssl_dir, openssl_cafile = os.path.split(
        ssl.get_default_verify_paths().openssl_cafile)

    print(" -- pip install --upgrade certifi")
    subprocess.check_call([sys.executable,
                           "-E", "-s", "-m", "pip", "install", "--upgrade", "certifi"])

    import certifi

    # change working directory to the default SSL directory
    os.chdir(openssl_dir)
    print(" -- removing any existing file or link")
    try:
        os.remove(openssl_cafile)
    except FileNotFoundError:
        pass
    print(" -- copying certifi certificate bundle")
    with open(certifi.where(), 'rb') as source_file:
        with open(openssl_cafile, 'wb') as dest_file:
            dest_file.write(source_file.read())
    print(" -- update complete")


if __name__ == '__main__':
    main()
