########### Python 3.2 #############
import urllib.request, json

try:
    url = "https://my-ur-fleet-production-api-management.azure-api.net/v3.4/openapi/robots/get-latest-log/1a921bad-fd66-ec11-8f8f-000d3a2e7278"

    hdr ={
    # Request headers
    'Cache-Control': 'no-cache',
    'Api-Key': '85c92e77cd734a7eb00101aae0200a36',
    }

    req = urllib.request.Request(url, headers=hdr)

    req.get_method = lambda: 'GET'
    response = urllib.request.urlopen(req)
    print(response.getcode())
    print(response.read())
except Exception as e:
    print(e)
####################################