# MDF in WebGME
This contains a tool for converting the
[MDF specification](https://github.com/ModECI/MDF/blob/documentation/docs/MDF_specification.json) into JSON
compatible with [JSON importer](https://github.com/deepforge-dev/webgme-json-importer/tree/master/src/common).
This allows us to programmatically create a metamodel and, as a result, use WebGME as a design environment for MDF.

## Quick Start

### Starting WebGME app
First, install the mdf_gme following:
- [NodeJS](https://nodejs.org/en/) (LTS recommended)
- [MongoDB](https://www.mongodb.com/)

Second, start mongodb locally by running the `mongod` executable in your mongodb installation
(you may need to create a `data` directory or set `--dbpath`).

Then, run `webgme start` from the project root to start . Finally, navigate to `http://localhost:8888` to start using
mdf_gme!

### Loading the spec into WebGME
First, install dependencies with `npm install`. Then convert the MDF specification using
```
node spec_to_gme.js path/to/MDF/spec.json
```

Finally, import the JSON into WebGME just like the
[examples](https://github.com/deepforge-dev/webgme-json-importer/tree/master/examples) (suffixed with "\_meta")!

### Loading instances to and from WebGME importable JSON and MDF
```
node bin/instance_converter path/to/MDForGME/instance.json
```
# Interactions between NeuroML and WebGME
