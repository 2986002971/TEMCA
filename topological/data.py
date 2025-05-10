"""Data structures for persistent homology calculations."""

from collections import namedtuple
from itertools import chain
from operator import itemgetter

import torch


def nesting_level(x):
    """Calculate nesting level of a list of objects.

    To convert between sparse and dense representations of topological
    features, we need to determine the nesting level of an input list.
    The nesting level is defined as the maximum number of times we can
    recurse into the object while still obtaining lists.

    Parameters
    ----------
    x : list
        Input list of objects.

    Returns
    -------
    int
        Nesting level of `x`. If `x` has no well-defined nesting level,
        for example because `x` is not a list of something, will return
        `0`.

    Notes
    -----
    This function is implemented recursively. It is therefore a bad idea
    to apply it to objects with an extremely high nesting level.

    Examples
    --------
    >>> nesting_level([1, 2, 3])
    1

    >>> nesting_level([[1, 2], [3, 4]])
    2
    """
    # This is really only supposed to work with lists. Anything fancier,
    # for example a `torch.tensor`, can already be used as a dense data
    # structure.
    if not isinstance(x, list):
        return 0

    # Empty lists have a nesting level of 1.
    if len(x) == 0:
        return 1
    else:
        return max(nesting_level(y) for y in x) + 1


class PersistenceInformation(
    namedtuple(
        "PersistenceInformation",
        [
            "pairing",
            "diagram",
            "dimension",
        ],
        # Ensures that there is always a dimension specified, albeit an
        # 'incorrect' one.
        defaults=[None],
    )
):
    """Persistence information data structure.

    This is a light-weight data structure for carrying information about
    the calculation of persistent homology. It consists of the following
    components:

    - A *persistence pairing*
    - A *persistence diagram*
    - An (optional) *dimension*

    Due to its lightweight nature, no validity checks are performed, but
    all calculation modules should return a sequence of instances of the
    :class:`PersistenceInformation` class.

    Since this data class is shared with modules that are capable of
    calculating persistent homology, the exact form of the persistence
    pairing might change. Please refer to the respective classes for
    more documentation.
    """

    __slots__ = ()

    # Disable iterating over the class since it collates heterogeneous
    # information and should rather be treated as a building block.
    __iter__ = None


def make_tensor_from_persistence_information(pers_info, extract_generators=False):
    """Convert (sequence) of persistence information entries to tensor.

    This function converts instance(s) of :class:`PersistenceInformation`
    objects into a single tensor. No padding will be performed. A client
    may specify what type of information to extract from the object. For
    instance, by default, the function will extract persistence diagrams
    but this behaviour can be changed by setting `extract_generators` to
    `true`.

    Parameters
    ----------
    pers_info : :class:`PersistenceInformation` or iterable thereof
        Input persistence information object(s). The function is able to
        handle both single objects and sequences. This has no bearing on
        the length of the returned tensor.

    extract_generators : bool
        If set, extracts generators instead of persistence diagram from
        `pers_info`.

    Returns
    -------
    torch.tensor
        Tensor of shape `(n, 3)`, where `n` is the sum of all features,
        over all dimensions in the input `pers_info`. Each triple shall
        be of the form `(creation, destruction, dim)` for a persistence
        diagram. If the client requested generators to be returned, the
        first two entries of the triple refer to *indices* with respect
        to the input data set. Depending on the algorithm employed, the
        meaning of these indices can change. Please refer to the module
        used to calculate persistent homology for more details.
    """
    # Looks a little bit cumbersome, but since `namedtuple` is iterable
    # as well, we need to ensure that we are actually dealing with more
    # than one instance here.
    if len(pers_info) > 1 and not isinstance(pers_info[0], PersistenceInformation):
        pers_info = [pers_info]

    # TODO: This might not always work since the size of generators
    # changes in different dimensions.
    if extract_generators:
        pairs = torch.cat(
            [torch.as_tensor(x.pairing, dtype=torch.float) for x in pers_info],
        ).long()
    else:
        pairs = torch.cat(
            [torch.as_tensor(x.diagram, dtype=torch.float) for x in pers_info],
        ).float()

    dimensions = torch.cat(
        [
            torch.as_tensor(
                [x.dimension] * len(x.diagram),
                dtype=torch.long,
                device=x.diagram.device,
            )
            for x in pers_info
        ]
    )

    result = torch.column_stack((pairs, dimensions))
    return result


def make_tensor(x):
    """Create dense tensor representation from sparse inputs.

    This function turns sparse inputs of :class:`PersistenceInformation`
    objects into 'dense' tensor representations, thus providing a useful
    integration into differentiable layers.

    The dimension of the resulting tensor depends on maximum number of
    topological features, summed over all dimensions in the data. This
    is similar to the format in `giotto-ph`.

    Parameters
    ----------
    x : list of (list of ...) :class:`PersistenceInformation`
        Input, consisting of a (potentially) nested list of
        :class:`PersistenceInformation` objects as obtained
        from a persistent homology calculation module, such
        as :class:`VietorisRipsComplex`.

    Returns
    -------
    torch.tensor
        Dense tensor representation of `x`. The output is best
        understood by considering some examples: given a batch
        obtained from :class:`VietorisRipsComplex`, our tensor
        will have shape `(B, N, 3)`. `B` is the batch size and
        `N` is the sum of maximum lengths of diagrams relative
        to this batch. Each entry will consist of a creator, a
        destroyer, and a dimension. Dummy entries, used to pad
        the batch, can be detected as `torch.nan`.
    """
    level = nesting_level(x)

    # Internal utility function for calculating the length of the output
    # tensor. This is required to ensure that all inputs can be *merged*
    # into a single output tensor.
    def _calculate_length(x, level):
        # Simple base case; should never occur in practice but let's be
        # consistent here.
        if len(x) == 0:
            return 0

        # Each `chain.from_iterable()` removes an additional layer of
        # nesting. We only have to start from level 2; we get level 1
        # for free because we can always iterate over a list.
        for i in range(2, level + 1):
            x = chain.from_iterable(x)

        # Collect information that we need to create the full tensor. An
        # entry of the resulting list contains the length of the diagram
        # and the dimension, making it possible to derive padding values
        # for all entries.
        M = list(map(lambda a: (len(a.diagram), a.dimension), x))

        # Get maximum dimension
        dim = max(M, key=itemgetter(1))[1]

        # Get *sum* of maximum number of entries for each dimension.
        # This is calculated over all batches.
        N = sum(
            [
                max([L for L in M if L[1] == d], key=itemgetter(0))[0]
                for d in range(dim + 1)
            ]
        )

        return N

    # Auxiliary function for padding tensors with `torch.nan` to
    # a specific dimension. Will always return a `list`; we turn
    # it into a tensor depending on the call level.
    def _pad_tensors(tensors, N, value=torch.nan):
        return list(
            map(
                lambda t: torch.nn.functional.pad(
                    t, (0, 0, N - len(t), 0), mode="constant", value=value
                ),
                tensors,
            )
        )

    N = _calculate_length(x, level)

    # List of lists: the first axis is treated as the batch axis, while
    # the second axis is treated as the dimension of diagrams or pairs.
    # This also handles ordinary lists, which will result in a batch of
    # size 1.
    if level <= 2:
        tensors = [
            make_tensor_from_persistence_information(pers_infos) for pers_infos in x
        ]

        # Pad all tensors to length N in the first dimension, then turn
        # them into a batch.
        result = torch.stack(_pad_tensors(tensors, N))
        return result

    # List of lists of lists: this indicates image-based data, where we
    # also have a set of tensors for each channel. The internal layout,
    # i.e. our input, has the following structure:
    #
    # B x C x D
    #
    # Each variable being the length of the respective list. We want an
    # output of the following shape:
    #
    # B x C x N x 3
    #
    # Here, `N` is the maximum length of an individual persistence
    # information object.
    else:
        tensors = [
            [
                make_tensor_from_persistence_information(pers_infos)
                for pers_infos in batch
            ]
            for batch in x
        ]

        # Pad all tensors to length N in the first dimension, then turn
        # them into a batch. We first stack over channels (inner), then
        # over the batch (outer).
        result = torch.stack(
            [torch.stack(_pad_tensors(batch_tensors, N)) for batch_tensors in tensors]
        )

        return result
