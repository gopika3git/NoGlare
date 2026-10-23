def load_labelmap(path):
    return {}

def convert_label_map_to_categories(label_map,
                                    max_num_classes,
                                    use_display_name=True):

    return [{'id': 1, 'name': 'vehicle'}]

def create_category_index(categories):

    return {1: {'id': 1, 'name': 'vehicle'}}
