const json = require(process.argv[2]);
const merge = require('lodash.merge');

function createMetaNode(name, data) {
    const parameters = parseParameters(data.allowed_parameters);
    const childrenMeta = data.allowed_children ? parseChildren(data.allowed_children) : {};
    const nodeDef = {
        id: `@meta:${name}`,
        attributes: {
            name,
            definition: data.definition,
        },
    };

    if (name.toLowerCase().includes('port')) {
        nodeDef.registry = {isPort: true};
    }

    return merge(nodeDef, parameters, childrenMeta);
}

function parseParameters(params) {
    const nodeDef = {};
    const paramEntries = Object.entries(params);
    paramEntries.forEach(entry => {
        const [name, desc] = entry;
        const referenceType = getReferenceType(desc);
        if (referenceType) {
            nodeDef.pointer_meta = nodeDef.pointer_meta || {};
            const ptrName = name.includes('sender') ? 'src' : 'dst';

            nodeDef.pointer_meta[ptrName] = nodeDef.pointer_meta[ptrName] || {min: -1, max: 1};
            nodeDef.pointer_meta[ptrName][`@meta:${referenceType}`] = {min: -1, max: 1};
        } else if (desc.type === 'str') {
            nodeDef.attribute_meta = nodeDef.attribute_meta || {};
            nodeDef.attribute_meta[name] = {type: 'string'};
        } else if (desc.type === 'dict') {
            nodeDef.children_meta = nodeDef.children_meta || {min: -1, max: -1};
            nodeDef.children_meta['@meta:DictionaryEntry'] = {min: -1, max: -1};

            nodeDef.pointer_meta = nodeDef.pointer_meta || {};
            nodeDef.pointer_meta[name] = {
                '@meta:DictionaryEntry': {min: -1, max: -1},
                min: -1,
                max: -1
            };
        } else {
            throw new Error(`Unsupported parameter type: Not sure how to convert ${desc.type} to GME`);
        }
    });

    return nodeDef;
}

function getReferenceType(desc) {
    // This is a little hacky :(
    const targetRegex = /The id of the _([a-zA-Z]+)_/;
    const match = desc.description.match(targetRegex);
    if (match) {
        return match[1];
    }
    return null;
}

function parseChildren(params) {
    // TODO: Do we need the names from the MDF format? I don't think so...
    const childTypes = Object.values(params).map(desc => desc.type);
    const childEntries = childTypes.map(type => [`@meta:${type}`, {min: -1, max: -1}]);
    return {
        children_meta: merge({
            min: -1,
            max: -1,
        }, Object.fromEntries(childEntries))
    };
}

const output = {
    attributes: {
        name: 'Language',
        version: json.version
    },
    children: Object.entries(json.specification)
        .map(entry => createMetaNode(...entry))
        .concat({
            id: `@meta:DictionaryEntry`,
            attribute_meta: {
                value: {type: 'string'}
            }
        })
};

console.log(JSON.stringify(output, null, 2));
