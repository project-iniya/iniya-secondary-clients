from typing import Literal, Optional, List, Dict, Any
from dataclasses import dataclass, field
from .utils import to_camel_case, get_device_id
from .Auth import verify_token, logout
import keyring
import requests
import io

# ─────────────────────────────────────────────────────────────────────────────
#  Types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractOptions:
    includeImages: Optional[bool] = None
    extractDepth: Optional[Literal["basic", "advanced"]] = None
    format: Optional[Literal["markdown", "text"]] = None
    timeout: Optional[int] = None
    includeFavicon: Optional[bool] = None
    includeUsage: Optional[bool] = None
    query: Optional[str] = None
    chunksPerSource: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
#  Mixin
# ─────────────────────────────────────────────────────────────────────────────

class SearchMixin:
    
    def setup_search_client(self, base_url: str = "https://iniyaai-backend.onrender.com/api/apis") -> None:
        self.base_url = base_url
        self.token = None
        self.initializeToken()
        

    def _post(self, endpoint: str, data: dict):
        # convert snake_case → camelCase
        camel_data = to_camel_case(data)

        res = requests.post(
            f"{self.base_url}/{endpoint}",
            json=camel_data,
            headers={
                "Authorization": f"Bearer {self.token}"
            }
        )

        res.raise_for_status()
        return res.json()

    def _get(self, endpoint: str, params: dict):
        camel_params = to_camel_case(params)
        print(camel_params)

        if self.token is None:
            raise Exception("No token found. Please login first.")

        res = requests.get(
            f"{self.base_url}/{endpoint}",
            params=camel_params,
            headers={
                "Authorization": f"Bearer {self.token}"
            }
        )

        res.raise_for_status()
        return res.json()
    
    def initializeToken(self):
        try:
          devid = get_device_id()
          if devid :
            token = keyring.get_password("IniyaAI", devid)
            if verify_token(token):
                self.token = token
            else:
                logout()
                raise Exception("Invalid Token, Logging Out")
          else:
              raise Exception("DEV ID Not Found")
        except Exception as e:
            print(e)
        finally:
            self.token = None

# Tavily-like function
    def search(
            self,
            query: str,
            search_depth: str = "basic",
            max_results: int = 5,
            include_domains: Optional[List[str]] = None,
            exclude_domains: Optional[List[str]] = None,
            include_answer: bool = False,
            include_raw_content: bool = False,
            **kwargs
    ) -> Dict[str, Any]:
        
        data = {
            "text": query,
            "func": "search",
            "options": {
                "search_depth": search_depth,
                "max_results": max_results,
                "include_domains": include_domains or [],
                "exclude_domains": exclude_domains or [],
                "include_answer": include_answer,
                "include_raw_content": include_raw_content,
                **kwargs
            }
        }

        return self._post("tavily", data)
    
    def extract(self, urls: List[str], options: ExtractOptions ) -> Dict[str, Any]:
        
        options_dict = {
            k: v for k, v in vars(options).items()
            if v is not None and k != "extra"
        }
        options_dict.update(options.extra)

        data = {
            "text":urls,
            "func":"extract",
            "options": options_dict
        }

        return self._post("tavily", data)