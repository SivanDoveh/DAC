import torch


def clear_pad_texts(text):
    clear_text_list = []
    list_amount_of_pos = []
    for t in text:
        non_zero_idx = t.sum(dim=1).nonzero()
        clear_text_list.append(t[non_zero_idx].squeeze(dim=1))
        list_amount_of_pos.append(len(non_zero_idx))
    try:
        cat = torch.cat(clear_text_list)
    except:
        print("l")
    return cat, list_amount_of_pos


def clear_pad(match_list, poss):
    clear_match_list = []
    clear_poss_list = []
    list_amount_of_pos = []
    for idx, row in enumerate(match_list):
        non_zero_idx = row.nonzero()
        clear_match_list.append(row[non_zero_idx])
        clear_poss_list.append(poss[idx][non_zero_idx])
        list_amount_of_pos.append(len(non_zero_idx))
    return clear_match_list, clear_poss_list, list_amount_of_pos


def average_pos_feat(text_features, list_amount_of_pos, match_list):
    device = text_features.device

    list_of_avg_pos = []
    pos_feats = text_features[: sum(list_amount_of_pos)]
    text_features = text_features[sum(list_amount_of_pos) :]
    last_index = 0
    for num_of_sentences, match_list_item in zip(list_amount_of_pos, match_list):
        list_of_avg_pos.append(
            sum(
                match_list_item.to(device, non_blocking=True, dtype=torch.float32)
                * pos_feats[last_index : last_index + num_of_sentences]
            )
            / num_of_sentences
        )
        last_index = last_index + num_of_sentences
    text_features = torch.cat((torch.stack(list_of_avg_pos), text_features), dim=0)
    return text_features
