import ast
from DFG import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from utils import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index,
                   extract_dataflow)
from tree_sitter import Language, Parser

dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'javascript':DFG_javascript
}


parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('./my-languages1.so', lang)   # Probablemente cambiarlo en linux a ./my-languages.so?
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser

def extract_ast_from_code(source_code):
    return ast.parse(source_code)

# provienen de convert_examples_to_features
def string_code_to_sequence(source_code):
    code_tokens,dfg = extract_dataflow(source_code, parser,"python")
    #code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
    #ori2cur_pos={}
    #ori2cur_pos[-1]=(0,0)
    #for i in range(len(code_tokens)):
    #    ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
    #code_tokens=[y for x in code_tokens for y in x]  
    ##truncating
    #code_tokens=code_tokens[:args.code_length+args.data_flow_length-2-min(len(dfg),args.data_flow_length)]
    #code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    #code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
    #position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
    #dfg=dfg[:args.code_length+args.data_flow_length-len(code_tokens)]
    #code_tokens+=[x[0] for x in dfg]
    #position_idx+=[0 for x in dfg]
    #code_ids+=[tokenizer.unk_token_id for x in dfg]
    #padding_length=args.code_length+args.data_flow_length-len(code_ids)
    #position_idx+=[tokenizer.pad_token_id]*padding_length
    #code_ids+=[tokenizer.pad_token_id]*padding_length      
    return code_tokens

def string_nl_to_sequence(source_nl):
    #nl_tokens=tokenizer.tokenize(nl)[:args.nl_length-2]
    #nl_tokens =[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    #nl_ids =  tokenizer.convert_tokens_to_ids(nl_tokens)
    #padding_length = args.nl_length - len(nl_ids)
    #nl_ids+=[tokenizer.pad_token_id]*padding_length
    return   


def extract_code_snippets(json_data, max_length=256):
        code_snippets = []

        def add_snippet(code):
            for i in range(0, len(code), max_length):
                code_snippets.append(code[i:i + max_length])

        def process_dict(a_dict):
            for key, value in a_dict.items():
                if isinstance(value, dict):
                    if 'code' in value:
                        add_snippet(value['code'])
                        value['code_sequence'] = string_code_to_sequence(value['code'])
                        #value['code_embedding'] = funcion que calcule embedding de code_sequence asumo? value['code'])
                    else:
                        process_dict(value)

        process_dict(json_data)
        return code_snippets    