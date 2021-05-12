/*globals define*/
/*eslint-env node, browser*/

define([
    'text!./metadata.json',
    'mdf_gme/instance-converter',
    'webgme-json-importer/JSONImporter',
    'plugin/PluginBase',
    'text!./template.py.ejs',
    'underscore',
], function (
    pluginMetadata,
    MDFConverter,
    JSONImporter,
    PluginBase,
    PythonCodeTpl,
    _,
) {
    'use strict';

    pluginMetadata = JSON.parse(pluginMetadata);
    PythonCodeTpl = _.template(PythonCodeTpl);

    class ExportToMDFPython extends PluginBase {
        constructor() {
            super();
            this.pluginMetadata = pluginMetadata;
        }

        async main(callback) {
            const mdfJson = await this.getMDFJson(this.activeNode);
            const code = PythonCodeTpl({mdfJson});
            const hash = await this.blobClient.putFile('output.py', code);
            this.result.addArtifact(hash);
            this.result.setSuccess(true);
            callback(null, this.result);
        }

        async getMDFJson(node) {
            const importer = new JSONImporter(this.core, this.rootNode);
            const json = await importer.toJSON(this.activeNode);
            await this.setBasePtrsToMetaTag(json);
            return Object.fromEntries([MDFConverter.Model.toMDF(json)]);
        }

        async setBasePtrsToMetaTag(json) {
            const {base} = json.pointers;
            const baseNode = await this.core.loadByPath(this.rootNode, base);
            const metaTag = `@meta:${this.core.getAttribute(baseNode, 'name')}`;
            json.pointers.base = metaTag;

            if (json.children) {
                json.children = await Promise.all(
                    json.children.map(child => this.setBasePtrsToMetaTag(child))
                );
            }
            return json;
        }
    }

    ExportToMDFPython.metadata = pluginMetadata;

    return ExportToMDFPython;
});
