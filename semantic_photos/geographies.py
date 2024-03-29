from typing import Dict, Any
from urllib.parse import urljoin

from urllib3.util.retry import Retry
import requests


class GeonamesAuthenticationError(Exception):
    ...


class GeonamesReverseGeocoder:
    """Convenience wrapper to some of the reverse geo-coding APIs from Geonames.
    See https://www.geonames.org/export/web-services.html for more details.

    Parameters
    ----------
    geonames_user : str
        Geonames API username for authentication
    """
    BASE_URL = "http://api.geonames.org"
    PRECISION = 3

    # See https://www.geonames.org/export/codes.html
    FEATURE_CODES = (
        "PRK",
        "RGN",
        "PPL",
        "MUS",
        "PAL",
        "GDN",
        "SQR",
        "SCH",
        "UNIV",
        "REST",
        "BCH"
    )

    def __init__(self, geonames_user: str):
        if not geonames_user:
            raise GeonamesAuthenticationError(
                "You must supply a username for the Geonames API. "
                "See: https://www.geonames.org/export/web-services.html"
            )

        retry_strategy = Retry(
            total=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        self.__session = requests.Session()
        self.__session.mount("https://", adapter)
        self.__session.mount("http://", adapter)

        self.user = geonames_user
        self.__cache = {}

    def _build_request(self, latitude: float, longitude: float) -> Dict[str, Any]:
        payload = {
            "username": self.user,
            "lat": latitude,
            "lng": longitude,
            "style": "FULL",
            "featureCode": self.FEATURE_CODES
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

    def _query(self, latitude: float, longitude: float, route: str) -> Dict[str, Any]:
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

    def find_nearby_place_name(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """Reverse geo-coding for nearby place names only.

        Parameters
        ----------
        latitude : float
        longitude : float

        Returns
        -------
        Dict[str, Any]
        """

        data = self._query(latitude=latitude, longitude=longitude, route="findNearbyPlaceNameJSON")
        return data

    def find_nearby(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """Reverse geo-coding for nearby places or points of interest.

        Parameters
        ----------
        latitude : float
        longitude : float

        Returns
        -------
        Dict[str, Any]
        """

        data = self._query(latitude=latitude, longitude=longitude, route="findNearbyJSON")
        return data

    def teardown(self):
        """Deletes cache and closes the session.
        """

        self.__cache.clear()
        self.__session.close()
