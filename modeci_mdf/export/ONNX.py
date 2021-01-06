'''
Simple export of MDF to ONNX... 

Work in progress...

'''

import sys
import onnx


def mdf_to_onnx(graph, save_to=None):
    
    print('Converting graph: %s to ONNX'%(graph.id))
    onodes = []
    oinputs = []
    ooutputs = []
    for node in graph.nodes:
        print('    Node: %s'%node.id)
        
        n = onnx.helper.make_node(
            'Sigmoid',
            name=node.id,
            inputs=[ip.id for ip in node.input_ports],
            outputs=[op.id for op in node.output_ports],
            )
        onodes.append(n)
        
    # Much more todo...   
    
    
    
    
            
    graph = onnx.helper.make_graph(
        nodes=onodes,
        name=graph.id,
        inputs=[],
        outputs=[])
        
    onnx_model = onnx.helper.make_model(graph)

    print('Model: %s'%onnx_model)
    if save_to:
        
        onnx.save(onnx_model, save_to)
        print('Saved ONNX to: %s'%save_to)
    
    return onnx_model
    
if __name__ == "__main__":
    
    from modeci_mdf.utils import load_mdf_json, print_summary
    
    example = '../../examples/Simple.json'
    verbose = True
    if len(sys.argv)==2:
        example = sys.argv[1]
        verbose = False
        
    mod_graph = load_mdf_json(example).graphs[0]
    
    print('Loaded Graph:')
    print_summary(mod_graph)
    
    print('------------------')
    
    mdf_to_onnx(mod_graph, save_to=example.replace('.json','.onnx'))
        