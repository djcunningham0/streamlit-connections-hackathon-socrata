from typing import Union, List, Generator

from streamlit.connections import ExperimentalBaseConnection
from streamlit import cache_data
from sodapy import Socrata
import pandas as pd


class SocrataConnection(ExperimentalBaseConnection):

    def _connect(self, **kwargs) -> Socrata:

        def get_param(x):
            """First retrieve value from kwargs, then from Streamlit secrets"""
            return kwargs.pop(x, self._secrets.get(x))

        return Socrata(
            domain=get_param("domain"),
            app_token=get_param("app_token"),
            username=get_param("username"),  # only required for modifying data
            password=get_param("password"),  # only required for modifying data
        )

    def client(self) -> Socrata:
        return self._instance

    def get(
            self,
            dataset_identifier: str,
            raw: bool = False,
            ttl: int = 3600,
            **kwargs,
    ) -> Union[pd.DataFrame, List[dict]]:

        @cache_data(ttl=ttl)
        def _get(dataset_identifier: str, raw: bool, **kwargs) -> Union[pd.DataFrame, List[dict]]:
            client = self.client()
            data = client.get(dataset_identifier=dataset_identifier, **kwargs)
            if raw:
                return data
            else:
                return pd.DataFrame.from_records(data)

        return _get(dataset_identifier=dataset_identifier, raw=raw, **kwargs)

    def get_all(
            self,
            dataset_identifier: str,
            raw: bool = False,
            ttl: int = 3600,
            **kwargs,
    ) -> Union[pd.DataFrame, Generator[dict, None, None]]:

        @cache_data(ttl=ttl)
        def _get_all(dataset_identifier: str, raw: bool, **kwargs) -> Union[pd.DataFrame, Generator[dict, None, None]]:
            client = self.client()
            data = client.get_all(dataset_identifier=dataset_identifier, **kwargs)
            if raw:
                return data
            else:
                return pd.DataFrame.from_records(data)

        return _get_all(dataset_identifier=dataset_identifier, raw=raw, **kwargs)
