def classFactory(iface):
    from .plugin import DeepLearningPlugin
    return DeepLearningPlugin(iface)