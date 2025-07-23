"""
This class is used for centralized configuration
"""

class Config:
    """
    Configuration data used all over the API.
    """

    _Config__conf = {
        "EMIS": {
            "token": "",  # Placeholder for EMIS Token
            "base_url": "https://api.emis.com"  # Add base_url for compatibility
        },
    }

    @staticmethod
    def get_emis_config():  # Getter for EMIS config
        return Config._Config__conf["EMIS"]
    
    # Add these methods for compatibility with app.py
    @classmethod
    def get(cls, section: str, key: str, default: str = "") -> str:
        """Get configuration value - for compatibility with app.py"""
        return cls._Config__conf.get(section, {}).get(key, default)
    
    @classmethod
    def set(cls, section: str, key: str, value: str) -> None:
        """Set configuration value - for compatibility with app.py"""
        if section not in cls._Config__conf:
            cls._Config__conf[section] = {}
        cls._Config__conf[section][key] = value