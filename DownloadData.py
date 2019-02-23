import urllib.request
import zipfile
print('Beginning file download with urllib2...')

url = 'https://www.dropbox.com/sh/c9n0ivfl82522au/AABgr8VZK3ya2aOkK7yWnylVa?dl=1'
urllib.request.urlretrieve(url, 'DL.zip')


zip_ref = zipfile.ZipFile('DL.zip', 'r')
zip_ref.extractall('.')
zip_ref.close()
zip_ref = zipfile.ZipFile('DB.zip', 'r')
zip_ref.extractall('.')
zip_ref.close()
zip_ref = zipfile.ZipFile('Save.zip', 'r')
zip_ref.extractall('.')
zip_ref.close()
