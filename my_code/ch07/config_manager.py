import os
import yaml

class ConfigManager:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self.load_config()

    def load_config(self):
        """加载配置文件"""
        config_path = os.path.join(
            os.path.dirname(__file__),
            'configs',
            'config.yaml'
        )
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        except Exception as e:
            print(f"加载配置文件失败: {str(e)}")
            self._config = {}

    @property
    def config(self):
        """获取配置"""
        return self._config

    @property
    def is_debug(self):
        """获取调试模式状态"""
        return self.config.get('settings', {}).get('debug', False)

# 创建全局配置管理器实例
config_manager = ConfigManager() 