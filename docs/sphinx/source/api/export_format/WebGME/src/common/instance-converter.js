function factory(_) {
    const Model = {
        toMDF(data) {
            const {name, format, generating_application, notes} = data.attributes;
            const graphs = Object.fromEntries(data.children.map(Graph.toMDF));
            const mdf = {
                format,
                generating_application,
                notes,
                graphs,
            };
            return [name, mdf];
        },

        toGME(name, data) {
            const {format, generating_application} = data;
            const children = Object.entries(data.graphs).map(entry => Graph.toGME(...entry));
            return {
                attributes: {
                    name,
                    id: name,
                    format,
                    generating_application,
                    notes: data.notes || '',
                },
                pointers: {
                    base: '@meta:Model',
                },
                children,
            };
        }
    };

    const Graph = {
        toMDF(data) {
            const {name, notes} = data.attributes;
            const {nodes=[], edges=[]} = _.groupBy(
                data.children,
                child => child.pointers.base.includes('Node') ? 'nodes' : 'edges'
            );
            const mdf = {
                notes,
                nodes: Object.fromEntries(nodes.map(Node.toMDF)),
                edges: Object.fromEntries(edges.map(edge => Edge.toMDF(nodes, edge))),
            };
            return [name, mdf];
        },

        toGME(name, data) {
            const children = Object.entries(data.nodes).map(entry => Node.toGME(...entry))
                .concat(Object.entries(data.edges).map(entry => Edge.toGME(...entry)))
            return {
                attributes: {
                    id: name,
                    name,
                    notes: data.notes || '',
                },
                pointers: {
                    base: '@meta:Graph',
                },
                children,
            };
        }
    };

    const Node = {
        toMDF(data) {
            const {name} = data.attributes;
            const mdf = {};

            const gmeBaseToMDFAttribute = {
                '@meta:DictionaryEntry': 'parameters',
                '@meta:InputPort': 'input_ports',
                '@meta:OutputPort': 'output_ports',
                '@meta:Function': 'functions',
            };
            const {parameters=[], input_ports=[], functions=[], output_ports=[]} = _.groupBy(
                data.children,
                child => {
                    if (gmeBaseToMDFAttribute[child.pointers.base]) {
                        return gmeBaseToMDFAttribute[child.pointers.base];
                    }
                    throw new Error(`Unrecognized base: ${child.pointers.base}`);
                }
            );
            if (parameters.length) {
                mdf.parameters = Object.fromEntries(parameters.map(Parameter.toMDF));
            }
            if (input_ports.length) {
                mdf.input_ports = Object.fromEntries(input_ports.map(InputPort.toMDF));
            }
            if (functions.length) {
                mdf.functions = Object.fromEntries(functions.map(Function.toMDF));
            }
            if (output_ports.length) {
                mdf.output_ports = Object.fromEntries(output_ports.map(OutputPort.toMDF));
            }
            return [name, mdf];
        },

        toGME(name, data) {
            const parameters = Object.entries(data.parameters).map(entry => Parameter.toGME(name, ...entry));

            const inputNodes = Object.entries(data.input_ports || {}).map(entry => InputPort.toGME(name, ...entry));
            const outputNodes = Object.entries(data.output_ports || {}).map(entry => OutputPort.toGME(name, ...entry));
            const functions = Object.entries(data.functions || {}).map(entry => Function.toGME(...entry));
            return {
                id: `@id:${name}`,
                attributes: {
                    name,
                    id: name,
                },
                pointers: {
                    base: '@meta:Node'
                },
                sets: {
                    parameters: parameters.map(param => param.id)
                },
                children: parameters.concat(inputNodes, outputNodes, functions)
            };
        }
    };

    const Parameter = {
        toMDF(data) {
            const {name, value} = data.attributes;
            return [name, value]
        },

        toGME(parentName, name, data) {
            return {
                id: `@id:${parentName}_${name}`,
                attributes: {
                    name,
                    value: data
                },
                pointers: {
                    base: '@meta:DictionaryEntry'
                }
            };
        }
    };

    const InputPort = {
        toMDF(data) {
            const {name, shape} = data.attributes;
            const mdf = {shape};
            return [name, mdf];
        },

        toGME(nodeName, name, data) {
            return {
                id: `@id:${nodeName}_${name}`,
                attributes: {
                    name,
                    id: name,
                    shape: data.shape,
                },
                pointers: {
                    base: '@meta:InputPort'
                },
            };
        }
    };

    const OutputPort = {
        toMDF(data) {
            const {name, value} = data.attributes;
            const mdf = {value}
            return [name, mdf];
        },

        toGME(nodeName, name, data) {
            return {
                id: `@id:${nodeName}_${name}`,
                attributes: {
                    name,
                    id: name,
                    value: data.value,
                },
                pointers: {
                    base: '@meta:OutputPort'
                }
            };
        }
    };

    const Function = {
        toMDF(data) {
            const {name} = data.attributes;
            const mdf = {
                function: data.attributes.function,
                args: Object.fromEntries(data.children.map(Parameter.toMDF)),
            };
            return [name, mdf];
        },

        toGME(name, data) {
            const args = Object.entries(data.args || {}).map(entry => Parameter.toGME(name, ...entry));
            return {
                attributes: {
                    name,
                    id: name,
                    function: data.function,
                    notes: data.notes || '',
                },
                pointers: {
                    base: '@meta:Function',
                },
                sets: {
                    args: args.map(arg => arg.id),
                },
                children: args,
            };
        }
    };

    const Edge = {
        toMDF(nodes, data) {
            const {name} = data.attributes;
            const portNodePairs = nodes
                .flatMap(
                    // TODO: use the @meta tags for meta pointers
                    node => node.children.filter(child => child.pointers.base.includes('Port'))
                        .map(port => [port, node])
                );
            const [sender_port, sender] = portNodePairs
                .find(pair => {
                    const [port, node] = pair;
                    return port.id === data.pointers.src || port.path === data.pointers.src;
                });
            const [receiver_port, receiver] = portNodePairs
                .find(pair => {
                    const [port, node] = pair;
                    return port.id === data.pointers.dst || port.path === data.pointers.dst;
                });
            const mdf = {
                sender: sender.attributes.name,
                receiver: receiver.attributes.name,
                sender_port: sender_port.attributes.name,
                receiver_port: receiver_port.attributes.name,
            };
            return [name, mdf];
        },

        toGME(name, data) {
            return {
                attributes: {name},
                pointers: {
                    base: '@meta:Edge',
                    src: `@id:${data.sender}_${data.sender_port}`,
                    dst: `@id:${data.receiver}_${data.receiver_port}`,
                }
            };
        }
    };

    return {Model};
}

if (typeof define !== 'undefined') {
    define(['underscore'], factory);
} else {
    module.exports = factory(require('underscore'));
}
