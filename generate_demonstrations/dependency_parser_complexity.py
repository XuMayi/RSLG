import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")

def dependency_analysis(text):
    # 使用spacy进行依存分析
    doc = nlp(text)
    #displacy.serve(doc, style='dep', auto_select_port=True)
    if len(list(doc.sents)) >= 2:
        return None, None, None

    def token_distance(token, root):
        distance = 0
        current_token = token
        while current_token != root:
            current_token = current_token.head
            distance += 1
        return distance

    rel = []
    for token in doc:
        if token.dep_ not in rel:
            rel.append(token.dep_)

    dep_count = len(rel)

    def calculate_depth(node):
        """递归计算依存树的深度"""
        if not list(node.children):
            return 0  # 叶子节点的深度为1
        else:
            return 1 + max(calculate_depth(child) for child in node.children)

    for token in doc:
        if token.dep_  == 'ROOT':
            max_dep = calculate_depth(token)
            root = token

    token_num = 0
    dep_all = 0
    for token in doc:
        token_num += 1
        if token.dep_ != 'ROOT':
            dep_all += token_distance(token,root)

    if token_num != 0:
        avg_depth = dep_all / token_num
    else:
        return None, None, None

    # for token in doc:
    #     print(
    #         '{0}({1}) <-- {2} -- {3}({4})'.format(token.text, token.tag_, token.dep_, token.head.text, token.head.tag_))


    return dep_count, avg_depth, max_dep


if __name__ == '__main__':
    #text = "I know he likes that beautiful girl."
    #Are Bocconia and Bellevalia both flowering plants?
 # What is the classification of Bocconia?
 # What is the classification of Bellevalia?
    text = 'Are Bocconia flowering plants?'
    result = dependency_analysis(text)
    print(result)

