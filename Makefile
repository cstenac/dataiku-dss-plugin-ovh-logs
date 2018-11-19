all:
	rm -rf dist
	# Make
	mkdir dist
	zip -r dist/dss-plugin-ovh-logs-import-0.0.3.zip plugin.json python-connectors
