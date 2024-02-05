from typing import Any
from urllib.parse import urljoin
import os

from urllib3.util.retry import Retry
import requests


class GeonamesReverseGeocoder:
    BASE_URL = "http://api.geonames.org"
    GEONAMES_USERNAME = os.environ["GEONAMES_USERNAME"]
    PRECISION = 3

    def __init__(self):
        retry_strategy = Retry(
            total=1,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS"]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        self.__session = requests.Session()
        self.__session.mount("https://", adapter)
        self.__session.mount("http://", adapter)

        self.__cache = {}

    @classmethod
    def _build_request(cls, latitude: float, longitude: float):
        payload = {
            "username": cls.GEONAMES_USERNAME,
            "lat": latitude,
            "lng": longitude
        }
        return payload

    def __check_cache(self, latitude: float, longitude: float, route: str):
        lat = round(latitude, self.PRECISION)
        lng = round(longitude, self.PRECISION)
        return self.__cache.get((lat, lng, route))

    def __upsert_cache(self, latitude: float, longitude: float, route: str, data: Any):
        lat = round(latitude, self.PRECISION)
        lng = round(longitude, self.PRECISION)
        self.__cache[(lat, lng, route)] = data

    def _query(self, latitude: float, longitude: float, route: str):
        cache_hit = self.__check_cache(latitude=latitude, longitude=longitude, route=route)
        if cache_hit is not None:
            return cache_hit

        response = self.__session.get(
            url=urljoin(self.BASE_URL, route),
            params=self._build_request(latitude=latitude, longitude=longitude)
        )

        if response.status_code == 200:
            data = response.json()
        else:
            data = {"geonames": []}
        self.__upsert_cache(latitude=latitude, longitude=longitude, route=route, data=data)
        return data


    def find_nearby_place_name(self, latitude: float, longitude: float):
        data = self._query(latitude=latitude, longitude=longitude, route="findNearbyPlaceNameJSON")
        return data

    def find_nearby(self, latitude: float, longitude: float):
        data = self._query(latitude=latitude, longitude=longitude, route="findNearbyJSON")
        return data

    def teardown(self):
        self.__cache.clear()
        self.__session.close()
