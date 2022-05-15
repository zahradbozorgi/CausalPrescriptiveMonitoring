from transformers.LastStateTransformer import LastStateTransformer
from transformers.AggregateTransformer import AggregateTransformer
        
def get_encoder(method, case_id_col=None, static_cat_cols=None, static_num_cols=None, dynamic_cat_cols=None, dynamic_num_cols=None, fillna=True, max_events=None):


    if method == "last":
        return LastStateTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols, fillna=fillna)

    elif method == "agg":
        return AggregateTransformer(case_id_col=case_id_col, cat_cols=dynamic_cat_cols, num_cols=dynamic_num_cols, boolean=False, fillna=fillna)

    else:
        print("Invalid encoder type")
        return None