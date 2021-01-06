'''
Simple export of MDF to ONNX... 
'''

import sys

def mdf_to_onnx(graph):
    
    print('Converting graph: %s to ONNX'%(graph.id))
    for node in graph.nodes:
        print('    Node: %s'%node.id)
        # todo...
    
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
    
    mdf_to_onnx(mod_graph)
        