import json

def get_sentence(fname):
    failed = False
    try:
        with open(fname, 'r') as f:
            item = json.load(f)
    except:
        print("Failed to load", fname)
        failed = True
    if not failed:
        original_prob = item['original prob']
        updated_prob = item['updated prob']
        original_pred = original_prob.index(max(original_prob))
        updated_pred = updated_prob.index(max(updated_prob))
        label = int(item['true label'])
        if (original_pred == label) and (updated_pred != original_pred):
            original = item['sentence']
            attack = item['updated sentence']
        else:
            return None, None
    else:
        return None, None
    return original, attack

def load_test_adapted_data_sentences(base_dir, num_test):
    '''
    Excludes data points with incorrect original predictions
    '''
    original_list_neg = []
    original_list_pos = []
    attack_list_neg = [] # Was originally negative
    attack_list_pos = [] # Was originally positive
    for i in range(num_test):
        fname = base_dir + '/neg'+str(i)+'.txt'
        original, attack = get_sentence(fname)
        if original is not None:
            original_list_neg.append(original)
            attack_list_neg.append(attack)

        fname = base_dir + '/pos'+str(i)+'.txt'
        original, attack = get_sentence(fname)
        if original is not None:
            original_list_pos.append(original)
            attack_list_pos.append(attack)

    return original_list_neg, original_list_pos, attack_list_neg, attack_list_pos