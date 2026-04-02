def classFactory(iface):
    from .plugin import GeoSegStudioPlugin
    return GeoSegStudioPlugin(iface)