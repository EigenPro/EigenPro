import logging
import numpy as np
import torch


alphabets_en = {
    'prot': np.array(
                ['R', 'H', 'K', 'D', 'E',
                 'S', 'T', 'N', 'Q', 'C',
                 'G', 'P', 'A', 'V', 'I',
                 'L', 'M', 'F', 'Y', 'W', ']']), #from MuE
    'prot_w_ins': np.array(
                ['R', 'H', 'K', 'D', 'E',
                 'S', 'T', 'N', 'Q', 'C',
                 'G', 'P', 'A', 'V', 'I',
                 'L', 'M', 'F', 'Y', 'W', '-', ']']),
    'dna': np.array(['A', 'C', 'G', 'T', ']']),
    'rna': np.array(['A', 'C', 'G', 'U', ']'])}


def get_ohe(seqs, alphabet_name='dna', include_stop=False, fill_stop=False):
    """Turn indicized or string sequences into OHEs.
    
    Parameters:
    seqs: numpy array of strs or ints
        seqs can have any number of dimensions, but must contain strings
        or be a list of indices of each letter in the alphabet. If seqs
        is already OHE, then returns seqs.
    alpahebt_name: str
        One of 'dna', 'rna', or 'prot'.
    include_stop: bool
        Add an extra dimension to the OHE for the end of the sequence.
        Sequences must include stop character ']' for a stop to be added
        (this is not checked).
    fill_stop: bool
        Fill all empty entries with stop symbols.
        
    Returns:
    ohes: numpy array
    """     
    alphabet = alphabets_en[alphabet_name]
    alphabet = alphabet[:len(alphabet) - 1 + include_stop]
    alphabet_size = len(alphabet)
    if isinstance(np.array(seqs).flatten()[0], str):
        # seqs is a set of strings
        max_seq_len = max([len(seq) for seq in np.array(seqs).flatten()])
        seq_shape = np.shape(seqs)
        seqs_t = np.array([np.pad(list(seq), (0, max_seq_len-len(seq)))
                           for seq in np.array(seqs).flatten()])
        ohe = seqs_t[..., None] == alphabet
        if fill_stop:
            assert include_stop, "Trying to fill stop without a stop column."
            ohe[..., -1] += (np.sum(ohe, axis=-1) == 0)
        return ohe.reshape(seq_shape + (max_seq_len, -1,))
    elif not np.all(seqs <= 1) and (not np.any(np.isnan(seqs))):
        # seqs is inds
        logging.warning("I would write a new ohe func if you're ohe'ing inds")
        ohe = seqs[..., None] == np.arange(alphabet_size)
        return np.nan_to_num(ohe, 0)
    else:
        # seqs is already ohe
        return seqs
    
def get_binarize(seqs, alphabet_name='dna', include_stop=False, fill_stop=False):
    """ Represents sequences as binary ints: empty spots are 1,
    the first letter is 10, second is 100, etc... . For easy
    Hamming distance calculation.
    
    Parameters:
    seqs: numpy array of strs or ints
        First put through get_ohe, accepts all representations except binary.
    alphabet_name: str
    include_stop: bool
    fill_stop: bool
    
    Returns:
    bin_seqs: numpy array
    """    
    ohe = get_ohe(seqs, alphabet_name=alphabet_name)
    alphabet_size = np.shape(ohe)[-1]
    return np.dot(ohe, 2**(np.arange(alphabet_size)+1)-1).astype(int) + 1


def get_inds(seqs, alphabet_name='dna', include_stop=False):
    """ Represents sequences as ints: empty spots are nan,
    the first letter is 0, second is 1, etc... . 
    
    Parameters:
    seqs: numpy array of strs or ints
        First put through get_ohe, accepts all representations except binary.
    alphabet_name: str
    include_stop: bool
        Include stop as a letter, distinct from empty spots.
    
    Returns:
    bin_seqs: numpy array
    """
    alphabet = alphabets_en[alphabet_name]
    alphabet = alphabet[:len(alphabet) - 1 + include_stop]
    alphabet_size = len(alphabet)
    inds = np.dot(get_ohe(
        seqs, alphabet_name=alphabet_name, include_stop=include_stop),
                  np.arange(alphabet_size, dtype=int)+1).astype(float)
    inds[inds == 0] = np.nan
    return inds - 1


def get_str(seqs, alphabet_name='dna', include_stop=False):
    """ Get string representation of OHE sequences. 
    
    Parameters:
    seqs: numpy array
    alphabet_name: str, default = 'dna'
    include_stop: bool, default = False
        Whether to incldue the stop in the alphabet.
        (Does the ohe have an extra row for stop?)
        
    Returns:
    strs: numpy array
        Has shape seqs.shape[:-2].
    """
    alphabet = alphabets_en[alphabet_name]
    alphabet = alphabet[:len(alphabet) - 1 + include_stop]
    seqs_flat = seqs.reshape((-1,) + np.shape(seqs)[-2:])
    lens = get_lens(seqs_flat).astype(int)
    strs = [''.join([alphabet[ohe][0] for ohe in seq[:len_].astype(bool)])
            for seq, len_ in zip(seqs_flat, lens)]
    return np.array(strs).reshape(np.shape(seqs)[:-2])


def get_lens(seqs_ohe):
    """ Get lengths of OHE sequences - numpy arrays or torch tensors. """
    return seqs_ohe.sum(axis=-1).sum(axis=-1)


################ Reformat axes of sequence representations #############


def add_stops(seqs, dtype=np.float64):
    """ Returns seqs with an extra index in the last dimension with 
    the stop symbol (only one stop is added - stop is not filled).
    An extra index in the length axis is also added, in case a sequence
    is as long as the ohe length axis. Returns nans for nan seq.
    
    Paramters:
    seqs: numpy array
        OHE sequences with any number of dimensions.
    
    Returns:
    seqs_w_stops: torch tensor
    """
    ohe_len = np.shape(seqs)[-2]
    seq_lens = get_lens(seqs)
    stops = (seq_lens[..., None] == np.arange(ohe_len + 1)).astype(dtype)
    stops[np.isnan(seq_lens), :] = np.nan
    seqs_w_stop = np.concatenate(
        [set_ohe_pad(seqs, 1, False), stops[..., None]], axis=-1).astype(dtype)
    return torch.tensor(seqs_w_stop)


def get_flat_seqs(seqs):
    """ Flattens all but the last two axes of OHE sequences.
    
    Parameters:
    seqs: numpy array
    
    Returns:
    flat_seqs: numpy array
    """
    return seqs.reshape([-1] + list(np.shape(seqs)[-2:]))


def pad_axis(seqs, pad_to_len, axis, pad_val, use_torch=False):
    """Pads axis of seqs to get their length to pad_to_len. If everything
    else in the pad axis is nan, pad is also nan.
    
    Parameters:
    seqs: numpy array or torch array
        Sequences represented not as str.
    pad_to_len: int
        Padded length of output.
    axis: int
        Axis to pad.
    pad_val: bool, int or float
        Value to pad with.
    use_torch: bool, default = False
        Whether seqs is a torch array.
        
    Returns:
    padded_seqs: numpy array
        Sequences with length pad_to_len.
    """
    shape = list(seqs.shape)
    if shape[axis] >= pad_to_len:
        index = [np.s_[:]] * len(seqs.shape)
        index[axis] = np.s_[pad_to_len:]
        if seqs[tuple(index)].sum() > 0:
            logging.warning("Sequence info lost by pad_seq_len.")
        index[axis] = np.s_[:pad_to_len]
        return seqs[tuple(index)]
    else:
        broadcast_index = [np.s_[:]] * len(seqs.shape)
        broadcast_index[axis] = np.s_[None]
        shape[axis] = pad_to_len - shape[axis]
        if use_torch:
            pad = pad_val * torch.ones(shape, dtype=seqs.dtype)
            axis_is_nan = torch.all(torch.isnan(seqs), axis=axis)
            axis_is_nan = axis_is_nan[broadcast_index].repeat_interleave(
                shape[axis], dim=axis)
            pad[axis_is_nan] = float('nan')
            return torch.cat([seqs, pad], axis=axis)
        else:
            pad = pad_val * np.ones(shape, dtype=seqs.dtype)
            axis_is_nan = np.all(np.isnan(seqs), axis=axis)
            axis_is_nan = axis_is_nan[tuple(broadcast_index)].repeat(
                shape[axis], axis=axis)
            if np.any(axis_is_nan):
                pad[axis_is_nan] = np.nan
            return np.concatenate([seqs, pad], axis=axis)


def pad_seq_len(seqs, pad_to_len):
    return pad_axis(seqs, pad_to_len, -2, 0)

def pad_seq_len_torch(seqs, pad_to_len):
    return pad_axis(seqs, pad_to_len, -2, 0, use_torch=True)

def pad_num_seqs(seqs, pad_to_len):
    return pad_axis(seqs, pad_to_len, -3, np.nan)


def set_ohe_pad(seqs, pad_len, use_max_len=True):
    """ Sets OHE pad for seqs.
    
    Parameters:
    seqs: numpy array
        Sequences represented not as str. Can be any number of dims.
    pad_len: int
        Padded length of output.
    use_max_len: bool, default = True
        If False, just add pads to current OHE pad.
        
    Returns:
    padded_seqs: numpy array
        Sequences with OHE length max_seq_len + pad_len.
    """
    if use_max_len:
        max_len = int(np.max(get_lens(seqs)))
    else:
        max_len = seqs.shape[-2]
    return pad_seq_len(seqs, max_len + pad_len)


def set_ohe_pad_ragged(seqs, pad_len, use_max_len=True):
    """ Sets OHE pad for seqs.
    
    Parameters:
    seqs: list of numpy array
        Sequences represented not as str. Can be any number of dims.
    pad_len: int
        Padded length of output.
    use_max_len: bool, default = True
        If False, just add pads to current OHE pad.
        
    Returns:
    padded_seqs: numpy array
        Sequences with OHE length max_seq_len + pad_len.
    """
    if use_max_len:
        max_len = int(np.max([get_lens(s) for s in seqs]))
    else:
        max_len = seqs.shape[-2]
    return np.array([pad_seq_len(s, max_len + pad_len) for s in seqs])


################ Hamming distances #################


def hamming_dist_slow(seqs_x, seqs_y, alphabet_name='dna', lag=1):
    """ Deprecated Hamming distance calculation using einsum. """
    ohe_x = get_ohe(seqs_x, alphabet_name=alphabet_name,
                              include_stop=True, fill_stop=True)
    ohe_y = get_ohe(seqs_y, alphabet_name=alphabet_name,
                              include_stop=True, fill_stop=True)
    shape_x = np.shape(ohe_x)[:-2]
    shape_y = np.shape(ohe_y)[:-2]
    length, alphabet_size = np.shape(ohe_x)[-2:]
    subscript_x = ''.join(['a', 'b', 'c', 'd', 'e'][:len(shape_x)])
    subscript_y = ''.join(['z', 'y', 'x', 'w', 'v'][:len(shape_y)])
    sims = np.einsum('{}lm,{}lm->{}{}l'.format(
        subscript_x, subscript_y, subscript_x, subscript_y),
                     ohe_x.astype(int), ohe_y.astype(int))
    difs = np.concatenate([1 - sims,
                           np.zeros(list(np.shape(sims)[:-1]) + [lag])], axis=-1)
    for lag_ in range(lag):
        if lag_>0:
            difs[..., :-lag_] += difs[..., lag_:]
    h_dist = np.sum(difs>=1, axis=-1)
    return h_dist


def hamming_dist(seqs_x, seqs_y, alphabet_name='dna', lag=1):
    """Calculate the hamming distance between two sets of sequences comparing
    k-mers of length lag at each position. Distances are calcualted as if
    sequences terminated with infinitely many stops.
    
    Parameters:
    seqs_x: numpy array of str or non-ragged array of one-hot encodings
        Can have any number of dimensions.
    seqs_y: numpy array of str or non-ragged array of one-hot encodings
        If in format of OHEs, must have length the same as seqs_x.
    alphabet_name: str
    lag: int
        length of k-mer comparisons.
        
    Returns:
    h_dists: numpy array
        Hamming distances of shape shape(seqs_x) + shape(seqs_y), ignoring
        the last two axes if sequences were OHE.
    """
    # if seqs_x is an array of str
    # first binarize each set of sequences to compare letters using bitwise_and
    if isinstance(seqs_x[0], str):
        bin_x = get_binarize(seqs_x, alphabet_name=alphabet_name,
                                    include_stop=True, fill_stop=True)
        bin_y = get_binarize(seqs_y, alphabet_name=alphabet_name,
                                    include_stop=True, fill_stop=True)
    else:
        bin_x = seqs_x
        bin_y = seqs_y
    shape_x = np.shape(bin_x)[:-1]
    shape_y = np.shape(bin_y)[:-1]
    lens = np.max([np.shape(bin_x)[-1], np.shape(bin_y)[-1]])
    bin_x = pad_axis(bin_x, lens, -1, 1).reshape([-1, lens])
    bin_y = pad_axis(bin_y, lens, -1, 1).reshape([-1, lens])
    h_dist= np.empty([len(bin_x), len(bin_y)])
    # cycle through batches of the first set of sequences and compare
    # (make sure largest mem alloc is not more than 1e8 entries)
    num_batches = np.ceil(len(bin_y) * len(bin_x) * lens / 1e8).astype(int)
    for batch_inds in np.array_split(np.arange(len(bin_x)),
                                     num_batches, axis=0):
        bin_x_b = bin_x[batch_inds]
        sims = np.bitwise_and(np.tile(bin_x_b[:, None, :], (1, len(bin_y), 1)),
                              np.tile(bin_y[None, :, :], (len(bin_x_b), 1, 1)))>0
        difs = 1 - sims
        for lag_ in range(lag):
            if lag_>0:
                difs[..., :-lag_] += difs[..., lag_:]
        h_dist[batch_inds] = np.sum(difs>=1, axis=-1)
    return h_dist.reshape(shape_x + shape_y)
