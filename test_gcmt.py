from eq_toolkit.sources.gcmt import get_events

catalog = get_events("gcmt.ndk")

print("Number of events:", len(catalog.data))
print(catalog.data.head())