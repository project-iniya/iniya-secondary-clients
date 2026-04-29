from .AudioMixin import AudioMixin
from .SearchMixin import SearchMixin
from .DeviceProfile import get_profile, ensure_models
from .Viz3DMixin import Viz3DMixin

class SearchClient(SearchMixin):
    def __init__(self, base_url: str = "https://iniyaai-backend.onrender.com/api/apis"):
        self.setup_search_client(base_url)

class AudioClient(AudioMixin):
    def __init__(self,verbose : bool = False):
        self.profile = get_profile(verbose=verbose)
        ensure_models(self.profile)
        self.setup_speech()

class VizualizerClient(Viz3DMixin):
    def __init__(self, start_server: bool = True, **kwargs):
        self.setup_3d(**kwargs)
        if start_server:
            self.serve_viewer()