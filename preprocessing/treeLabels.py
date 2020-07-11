import json


def tree2branch(file, thread_id):
    with open(file, 'r', encoding='utf-8') as f:
        tree_json = json.load(f)
        # print(len(tree_json[thread_id]))
        thread_list = list()
        for v in tree_json[thread_id]:
            value = tree_json[thread_id][v]
            branch_list = list()
            branch_list.append(thread_id)
            if isinstance(value, dict):
                branch_list.append(v)
                list_dictionary(value, branch_list)
            else:
                branch_list.append(v)
            thread_list.append(branch_list)
    return thread_list


def list_dictionary(dict, l):
    for k, v in dict.items():
        try:
            l.append(k)
            if v != []:
                list_dictionary(v, l)
        except AttributeError:
            print('wrong')
        break
    return l

# file = 'all-rnr-annotated-threads/charliehebdo-all-rnr-threads/rumours/552783238415265792/structure.json'
# t = tree2branch(file, '552783238415265792')
# print(t)
