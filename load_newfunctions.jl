# various new functions (or new methods for existing functions)

# we consider three different rules for selecting nodes to sample

# (1) sample off-the-tree (possibly) every Nth iteration
# (2) sample off-the-tree after any sufficiently big improvement to δ
# (3) sample off-the-tree if all the weighted oracle gaps are below A*exp(\alpha*k)
# (4) sample off-tree unless some scaled local improvement is above \delta*f(k) for some decaying function f(k)

## New methods for already-existing functions start here

function gen_B3_type(st::Int64,c::Int64,a::Int64,w::Int64,j::Int64,e::Float64)::B3_type

    ms,mp,ps,pp,unc = load_data(c)

    mp.π = zeros(length(mp.π));
    mp.π0 = zeros(length(mp.π0));

    if st == 0
        mp.π0[1] = 1.0;
        mp.π0[15] = 1.0;
        mp.π[14] = 1.0;
        mp.π[392] = 1.0;
    elseif st == 1
        mp.π0[1] = 1.0;
        mp.π0[14:16] = (1/3)*ones(3);
        mp.π[13:15] = (1/3)*ones(3);
        mp.π[364:366] = (1/9)*ones(3);
        mp.π[391:393] = (1/9)*ones(3);
        mp.π[418:420] = (1/9)*ones(3);
    elseif st == 2
        mp.π0[1] = 1.0;
        mp.π0[11:19] = (1/9)*ones(9);
        mp.π[10:18] = (1/9)*ones(9);
        mp.π[280:288] = (1/81)*ones(9);
        mp.π[307:315] = (1/81)*ones(9);
        mp.π[334:342] = (1/81)*ones(9);
        mp.π[361:369] = (1/81)*ones(9);
        mp.π[388:396] = (1/81)*ones(9);
        mp.π[415:423] = (1/81)*ones(9);
        mp.π[442:450] = (1/81)*ones(9);
        mp.π[469:477] = (1/81)*ones(9);
        mp.π[496:504] = (1/81)*ones(9);
    end

    d = gen_D__type(ms,mp,ps,pp,unc,c,a,.0,w,j,e)
    r = gen_R__type(d)
    t = gen_T3_type(d)
    h = gen_H__type(d)
    m = gen_M__type(d)

    return B3_type(r,t,h,m,d)
end

function set_Iex!(b::B3_type,N::Int64)::Tuple{B3_type,Bool}

    off_tree = false;

    if length(filter(!iszero,b.data.mp.π)) == 2
        more_π = zeros(756);

        bigger_tree = [5,11,13,14,15,17,23,
                       140,149,158,308,311,314,364,365,366,
                       383,389,391,392,393,395,401,
                       418,419,420,470,473,476,626,635,644];
        for bt in bigger_tree
            more_π[bt] = 1.0;
        end
    elseif length(filter(!iszero,b.data.mp.π)) == 12
        more_π = zeros(756);

        # can only visit nodes where one of the parameters takes mean values
        # these are hard-coded below (based on some Excel manipulations)

        if b.data.mp.π[6] != 0
            # existing tree is for c_ur free
            biggest_tree = [2,4,5,6,8,11,13,14,15,17,20,22,23,24,26,
                            56,59,62,65,68,71,74,77,80,112,113,114,121,122,
                            123,130,131,132,137,139,140,141,143,146,148,149,
                            150,152,155,157,158,159,161,166,167,168,175,176,
                            177,184,185,186,218,221,224,227,230,233,236,239,
                            242,299,302,305,308,311,314,317,320,323,355,356,
                            357,364,365,366,373,374,375,380,382,383,384,386,
                            389,391,392,393,395,398,400,401,402,404,409,410,
                            411,418,419,420,427,428,429,461,464,467,470,473,
                            476,479,482,485,542,545,548,551,554,557,560,563,
                            566,598,599,600,607,608,609,616,617,618,623,625,
                            626,627,629,632,634,635,636,638,641,643,644,645,
                            647,652,653,654,661,662,663,670,671,672,704,707,
                            710,713,716,719,722,725,728];
        elseif b.data.mp.π[12] != 0;
            # existing tree is for c_c02 free
            biggest_tree = [2,5,8,10,11,12,13,14,15,16,17,18,20,23,26,
                            56,59,62,65,68,71,74,77,80,137,140,143,146,149,
                            152,155,158,161,218,221,224,227,230,233,236,239,
                            242,280,281,282,283,284,285,286,287,288,299,302,
                            305,307,308,309,310,311,312,313,314,315,317,320,
                            323,334,335,336,337,338,339,340,341,342,361,362,
                            363,364,365,366,367,368,369,380,383,386,388,389,
                            390,391,392,393,394,395,396,398,401,404,415,416,
                            417,418,419,420,421,422,423,442,443,444,445,446,
                            447,448,449,450,461,464,467,469,470,471,472,473,
                            474,475,476,477,479,482,485,496,497,498,499,500,
                            501,502,503,504,542,545,548,551,554,557,560,563,
                            566,623,626,629,632,635,638,641,644,647,704,707,
                            710,713,716,719,722,725,728];
        else
            # existing tree is for v_CO2 free
            biggest_tree = [4,5,6,10,11,12,13,14,15,16,17,18,22,23,24,
                            112,113,114,121,122,123,130,131,132,139,140,141,
                            148,149,150,157,158,159,166,167,168,175,176,177,
                            184,185,186,280,281,282,283,284,285,286,287,288,
                            307,308,309,310,311,312,313,314,315,334,335,336,
                            337,338,339,340,341,342,355,356,357,361,362,363,
                            364,365,366,367,368,369,373,374,375,382,383,384,
                            388,389,390,391,392,393,394,395,396,400,401,402,
                            409,410,411,415,416,417,418,419,420,421,422,423,
                            427,428,429,442,443,444,445,446,447,448,449,450,
                            469,470,471,472,473,474,475,476,477,496,497,498,
                            499,500,501,502,503,504,598,599,600,607,608,609,
                            616,617,618,625,626,627,634,635,636,643,644,645,
                            652,653,654,661,662,663,670,671,672];
        end

        for bt in biggest_tree
            more_π[bt] = 1.0;
        end
    else
        more_π = ones(756);
    end

    for w in 1:b.data.w

        if (b.hist.k % N == 0)
            ie = findmax(more_π.*(b.temp.θu[b.data.kidx[w]].-b.temp.θl[b.data.kidx[w]]))[2];
            off_tree = true;
        else
            ie = findmax(b.data.mp.π[b.data.kidx[w]].*(b.temp.θu[b.data.kidx[w]].-b.temp.θl[b.data.kidx[w]]))[2];
        end

        b.temp.Ie[w] = b.data.iidx[w][ie];

        str = "Solving the subproblem at node $ie";
        str *= " "^(4-length("$ie"));
        print(str)
    end

    return (b,off_tree)
end

function set_Iex!(b::B3_type,γ::Float64)::Tuple{B3_type,Bool}

    off_tree = false;

    if length(filter(!iszero,b.data.mp.π)) == 2
        more_π = zeros(756);

        bigger_tree = [5,11,13,14,15,17,23,
                       140,149,158,308,311,314,364,365,366,
                       383,389,391,392,393,395,401,
                       418,419,420,470,473,476,626,635,644];
        for bt in bigger_tree
            more_π[bt] = 1.0;
        end
    elseif length(filter(!iszero,b.data.mp.π)) == 12
        more_π = zeros(756);

        # can only visit nodes where one of the parameters takes mean values
        # these are hard-coded below (based on some Excel manipulations)

        if b.data.mp.π[6] != 0
            # existing tree is for c_ur free
            biggest_tree = [2,4,5,6,8,11,13,14,15,17,20,22,23,24,26,
                            56,59,62,65,68,71,74,77,80,112,113,114,121,122,
                            123,130,131,132,137,139,140,141,143,146,148,149,
                            150,152,155,157,158,159,161,166,167,168,175,176,
                            177,184,185,186,218,221,224,227,230,233,236,239,
                            242,299,302,305,308,311,314,317,320,323,355,356,
                            357,364,365,366,373,374,375,380,382,383,384,386,
                            389,391,392,393,395,398,400,401,402,404,409,410,
                            411,418,419,420,427,428,429,461,464,467,470,473,
                            476,479,482,485,542,545,548,551,554,557,560,563,
                            566,598,599,600,607,608,609,616,617,618,623,625,
                            626,627,629,632,634,635,636,638,641,643,644,645,
                            647,652,653,654,661,662,663,670,671,672,704,707,
                            710,713,716,719,722,725,728];
        elseif b.data.mp.π[12] != 0;
            # existing tree is for c_c02 free
            biggest_tree = [2,5,8,10,11,12,13,14,15,16,17,18,20,23,26,
                            56,59,62,65,68,71,74,77,80,137,140,143,146,149,
                            152,155,158,161,218,221,224,227,230,233,236,239,
                            242,280,281,282,283,284,285,286,287,288,299,302,
                            305,307,308,309,310,311,312,313,314,315,317,320,
                            323,334,335,336,337,338,339,340,341,342,361,362,
                            363,364,365,366,367,368,369,380,383,386,388,389,
                            390,391,392,393,394,395,396,398,401,404,415,416,
                            417,418,419,420,421,422,423,442,443,444,445,446,
                            447,448,449,450,461,464,467,469,470,471,472,473,
                            474,475,476,477,479,482,485,496,497,498,499,500,
                            501,502,503,504,542,545,548,551,554,557,560,563,
                            566,623,626,629,632,635,638,641,644,647,704,707,
                            710,713,716,719,722,725,728];
        else
            # existing tree is for v_CO2 free
            biggest_tree = [4,5,6,10,11,12,13,14,15,16,17,18,22,23,24,
                            112,113,114,121,122,123,130,131,132,139,140,141,
                            148,149,150,157,158,159,166,167,168,175,176,177,
                            184,185,186,280,281,282,283,284,285,286,287,288,
                            307,308,309,310,311,312,313,314,315,334,335,336,
                            337,338,339,340,341,342,355,356,357,361,362,363,
                            364,365,366,367,368,369,373,374,375,382,383,384,
                            388,389,390,391,392,393,394,395,396,400,401,402,
                            409,410,411,415,416,417,418,419,420,421,422,423,
                            427,428,429,442,443,444,445,446,447,448,449,450,
                            469,470,471,472,473,474,475,476,477,496,497,498,
                            499,500,501,502,503,504,598,599,600,607,608,609,
                            616,617,618,625,626,627,634,635,636,643,644,645,
                            652,653,654,661,662,663,670,671,672];
        end

        for bt in biggest_tree
            more_π[bt] = 1.0;
        end
    else
        more_π = ones(756);
    end

    for w in 1:b.data.w

        if (b.hist.k > 2) && (b.hist.U[b.hist.k-1]-b.hist.L[b.hist.k-1])/(b.hist.U[b.hist.k-2] - b.hist.L[b.hist.k-2]) < γ
            ie = findmax(more_π.*(b.temp.θu[b.data.kidx[w]].-b.temp.θl[b.data.kidx[w]]))[2];
            off_tree = true;
        else
            ie = findmax(b.data.mp.π[b.data.kidx[w]].*(b.temp.θu[b.data.kidx[w]].-b.temp.θl[b.data.kidx[w]]))[2];
        end

        b.temp.Ie[w] = b.data.iidx[w][ie];

        str = "Solving the subproblem at node $ie";
        str *= " "^(4-length("$ie"));
        print(str)
    end

    return (b,off_tree)
end

function set_Iex!(b::B3_type,A::Float64,α::Float64)::Tuple{B3_type,Bool}

    off_tree = false;

    # we're sampling from a subtree T_0 contained in three either subtrees
    # T1, T2, T3 ... only want to sample from their union, not the full tree

    if length(filter(!iszero,b.data.mp.π)) == 2
        more_π = zeros(756);

        # can only visit nodes where two of the parameters take mean values
        # these are hard-coded below (based on some Excel manipulations)
        bigger_tree = [5,11,13,14,15,17,23,
                       140,149,158,308,311,314,364,365,366,
                       383,389,391,392,393,395,401,
                       418,419,420,470,473,476,626,635,644];
        for bt in bigger_tree
            more_π[bt] = 1.0;
        end
    elseif length(filter(!iszero,b.data.mp.π)) == 12
        more_π = zeros(756);

        # can only visit nodes where one of the parameters takes mean values
        # these are hard-coded below (based on some Excel manipulations)

        if b.data.mp.π[6] != 0
            # existing tree is for c_ur free
            biggest_tree = [2,4,5,6,8,11,13,14,15,17,20,22,23,24,26,
                            56,59,62,65,68,71,74,77,80,112,113,114,121,122,
                            123,130,131,132,137,139,140,141,143,146,148,149,
                            150,152,155,157,158,159,161,166,167,168,175,176,
                            177,184,185,186,218,221,224,227,230,233,236,239,
                            242,299,302,305,308,311,314,317,320,323,355,356,
                            357,364,365,366,373,374,375,380,382,383,384,386,
                            389,391,392,393,395,398,400,401,402,404,409,410,
                            411,418,419,420,427,428,429,461,464,467,470,473,
                            476,479,482,485,542,545,548,551,554,557,560,563,
                            566,598,599,600,607,608,609,616,617,618,623,625,
                            626,627,629,632,634,635,636,638,641,643,644,645,
                            647,652,653,654,661,662,663,670,671,672,704,707,
                            710,713,716,719,722,725,728];
        elseif b.data.mp.π[12] != 0;
            # existing tree is for c_c02 free
            biggest_tree = [2,5,8,10,11,12,13,14,15,16,17,18,20,23,26,
                            56,59,62,65,68,71,74,77,80,137,140,143,146,149,
                            152,155,158,161,218,221,224,227,230,233,236,239,
                            242,280,281,282,283,284,285,286,287,288,299,302,
                            305,307,308,309,310,311,312,313,314,315,317,320,
                            323,334,335,336,337,338,339,340,341,342,361,362,
                            363,364,365,366,367,368,369,380,383,386,388,389,
                            390,391,392,393,394,395,396,398,401,404,415,416,
                            417,418,419,420,421,422,423,442,443,444,445,446,
                            447,448,449,450,461,464,467,469,470,471,472,473,
                            474,475,476,477,479,482,485,496,497,498,499,500,
                            501,502,503,504,542,545,548,551,554,557,560,563,
                            566,623,626,629,632,635,638,641,644,647,704,707,
                            710,713,716,719,722,725,728];
        else
            # existing tree is for v_CO2 free
            biggest_tree = [4,5,6,10,11,12,13,14,15,16,17,18,22,23,24,
                            112,113,114,121,122,123,130,131,132,139,140,141,
                            148,149,150,157,158,159,166,167,168,175,176,177,
                            184,185,186,280,281,282,283,284,285,286,287,288,
                            307,308,309,310,311,312,313,314,315,334,335,336,
                            337,338,339,340,341,342,355,356,357,361,362,363,
                            364,365,366,367,368,369,373,374,375,382,383,384,
                            388,389,390,391,392,393,394,395,396,400,401,402,
                            409,410,411,415,416,417,418,419,420,421,422,423,
                            427,428,429,442,443,444,445,446,447,448,449,450,
                            469,470,471,472,473,474,475,476,477,496,497,498,
                            499,500,501,502,503,504,598,599,600,607,608,609,
                            616,617,618,625,626,627,634,635,636,643,644,645,
                            652,653,654,661,662,663,670,671,672];
        end

        for bt in biggest_tree
            more_π[bt] = 1.0;
        end
    else
        more_π = ones(756);
    end

    for w in 1:b.data.w
        (val,ie) = findmax(b.data.mp.π[b.data.kidx[w]].*(b.temp.θu[b.data.kidx[w]].-b.temp.θl[b.data.kidx[w]]));

        if (val <= A*exp(-α * b.hist.k))
            ie = findmax(more_π.*(b.temp.θu[b.data.kidx[w]].-b.temp.θl[b.data.kidx[w]]))[2];
            off_tree = true;
        end

        b.temp.Ie[w] = b.data.iidx[w][ie];

        str = "Solving the subproblem at node $ie";
        str *= " "^(4-length("$ie"));
        print(str)
    end


    return (b,off_tree)
end

function step_c!(b::B3_type,s::S3_type,N::Int64)::Tuple{B3_type,S3_type,Bool}

    off_tree = set_Iex!(b,N)[2];
    for s.temp.i in b.temp.Ie
        solv_exact!(s)
        update_s!(s)
    end

    return (b,s,off_tree)
end

function step_c!(b::B3_type,s::S3_type,γ::Float64)::Tuple{B3_type,S3_type,Bool}

    off_tree = set_Iex!(b,γ)[2];
    for s.temp.i in b.temp.Ie
        solv_exact!(s)
        update_s!(s)
    end

    return (b,s,off_tree)
end

function step_c!(b::B3_type,s::S3_type,A::Float64,α::Float64)::Tuple{B3_type,S3_type,Bool}

    off_tree = set_Iex!(b,A,α)[2];
    for s.temp.i in b.temp.Ie
        solv_exact!(s)
        update_s!(s)
    end

    return (b,s,off_tree)
end

function step!(b::B3_type,s::S3_type,N::Int64)::Tuple{B3_type,S3_type}

    b.hist.k += 1
    b.hist.T[b.hist.k,1] = @elapsed step_a!(b,s)
    b.hist.T[b.hist.k,2] = @elapsed step_b!(b)
    b.hist.T[b.hist.k,3] = @elapsed off_tree = step_c!(b,s,N)[3]
    b.hist.T[b.hist.k,4] = @elapsed step_d!(b,s)
    b.hist.T[b.hist.k,5] = @elapsed step_e!(b)
    b.hist.T[b.hist.k,6] = @elapsed step_f!(b)

    a1 = "$(round(b.temp.Δ;digits=3))" * "0" ^ (5 - length("$(round(b.temp.Δ%1;digits=3))"))
    a2 = "$(round(sum(b.hist.T[b.hist.k,:]);digits=2))"
    str = " k =" * (" " ^ (4-length("$(b.hist.k)"))) * "$(b.hist.k), "
    str *= "δ =" * (" " ^ (7-length(a1))) * "$(a1) %, "
    str *= "t =" * (" " ^ (8-length(a2))) * "$(a2) s"

    if off_tree
        str *= " (exploring off the tree this iteration)";
    end

    println(str)

    return b,s
end

function step!(b::B3_type,s::S3_type,γ::Float64)::Tuple{B3_type,S3_type}

    b.hist.k += 1
    b.hist.T[b.hist.k,1] = @elapsed step_a!(b,s)
    b.hist.T[b.hist.k,2] = @elapsed step_b!(b)
    b.hist.T[b.hist.k,3] = @elapsed off_tree = step_c!(b,s,γ)[3]
    b.hist.T[b.hist.k,4] = @elapsed step_d!(b,s)
    b.hist.T[b.hist.k,5] = @elapsed step_e!(b)
    b.hist.T[b.hist.k,6] = @elapsed step_f!(b)

    a1 = "$(round(b.temp.Δ;digits=3))" * "0" ^ (5 - length("$(round(b.temp.Δ%1;digits=3))"))
    a2 = "$(round(sum(b.hist.T[b.hist.k,:]);digits=2))"
    str = " k =" * (" " ^ (4-length("$(b.hist.k)"))) * "$(b.hist.k), "
    str *= "δ =" * (" " ^ (7-length(a1))) * "$(a1) %, "
    str *= "t =" * (" " ^ (8-length(a2))) * "$(a2) s"

    if off_tree
        str *= " (exploring off the tree this iteration)";
    end

    println(str)

    return b,s
end

function step!(b::B3_type,s::S3_type,A::Float64,α::Float64)::Tuple{B3_type,S3_type}

    b.hist.k += 1
    b.hist.T[b.hist.k,1] = @elapsed step_a!(b,s)
    b.hist.T[b.hist.k,2] = @elapsed step_b!(b)
    b.hist.T[b.hist.k,3] = @elapsed off_tree = step_c!(b,s,A,α)[3]
    b.hist.T[b.hist.k,4] = @elapsed step_d!(b,s)
    b.hist.T[b.hist.k,5] = @elapsed step_e!(b)
    b.hist.T[b.hist.k,6] = @elapsed step_f!(b)

    a1 = "$(round(b.temp.Δ;digits=3))" * "0" ^ (5 - length("$(round(b.temp.Δ%1;digits=3))"))
    a2 = "$(round(sum(b.hist.T[b.hist.k,:]);digits=2))"
    str = " k =" * (" " ^ (4-length("$(b.hist.k)"))) * "$(b.hist.k), "
    str *= "δ =" * (" " ^ (7-length(a1))) * "$(a1) %, "
    str *= "t =" * (" " ^ (8-length(a2))) * "$(a2) s"

    if off_tree
        str *= " (exploring off the tree this iteration)";
    end

    println(str)

    return b,s
end

## New functions start here

function get_subtree()::Int64

    println("")
    println("Choose one of the three subtrees below:")
    println(" subtree 0 -> 0 uncertain parameters")
    println(" subtree 1 -> 1 uncertain parameter")
    println(" subtree 2 -> 2 uncertain parameters")
    println("")
    print("Select subtree to explore:  ")
    cs = "";
    cs = readline()
    while (cs ∉ ["0","1","2"])
        println("Error: select 0, 1, or 2")
        print("Select subtree to explore:  ")
        cs = readline()
    end

    return parse(Int64,cs)
end

function get_Iexrule()::Int64

    println("")
    println("Choose one of the rules below:")
    println(" rule 1: explore off the subtree every N iterations ")
    println(" rule 2: explore off the subtree after making a big improvement to the global Δ")
    println(" rule 3: explore off the subtree if all local improvements at this iteration are smaller than some exponential function")
    println(" rule 4: explore off the subtree unless local improvements are smaller than Δ*f(k) some fixed function f")
    println("")
    print("Select rule to use:  ")
    cs = ""
    cs = readline()
    while (cs ∉ ["1","2","3","4"])
        println("Error: select 1, 2, 3 or 4")
        print("Select rule to use:  ")
        cs = readline()
    end

    return parse(Int64,cs)
end

function rule1_params()::Int64
    println("")
    print("Explore off the subtree once every how many iterations?  ")
    N = readline()

    return parse(Int64,N)
end

function rule2_params()::Float64
    println("")
    print("Explore off the subtree if Δ at last iteration was less than what fraction of the previous Δ?  ")
    γ = ""
    γ = readline()

    return parse(Float64,γ)
end

function rule3_params()::Tuple{Float64,Float64}
    println("")
    println("")
    print("What should the initial value of the threshold be?  ")
    A = ""
    A = readline()
    println("")
    print("And what should the decay rate be?  ")
    α = ""
    α = readline()

    return (parse(Float64,A), parse(Float64,α))
end

function print_summary_c(b::B3_type,c::Int64)

    println(" */"*"-"^ 32*"/*")
    println(" ")
    println(" */"*"-"^ 68*"/*")
    x0 = value.(b.rmp.m[:x0])
    x1 = round.(x0[:,1];digits=1)
    x2 = round.(x0[:,2:end];digits=1)
    str_tech = ["coal","coalccs","OCGT","CCGT","diesel","nuclear","pumpL","pumpH","lithium","onwind","offwind","solar"]
    int1 = " tech."*" "^9*"i1"
    println(" ")
    println(" co2 emission limit : $( c <= 0 ? "known" : "uncertain" )")
    println(" co2 emission cost  : $( c <= 1 ? "known" : "uncertain" )")
    println(" uranium cost       : $( c <= 2 ? "known" : "uncertain" )")
    println(" investment nodes   : $( b.data.ms.I0[end] ) (of which subtree includes $(1+3^c) )")
    println(" operational nodes  : $( b.data.ms.I[ end] ) (of which subtree includes $(3^c+9^c) )")
    println(" optimal objective  : $( round(b.hist.U[b.hist.k]*exp10(-11);digits=3) ) x 10^11 £")
    println(" ")
    println(" */"*"-"^ 68*"/*")
    println(" ")
    println(" optimal investments (GW) @ 0 years")
    println(" " * "-" ^ (length(int1)-1))
    println(int1)
    println(" " * "-" ^ (length(int1)-1))
    for p in b.data.ms.P
        println(str_fcn1(x1[p,:],str_tech[p]))
    end
    println(" "*"-"^(length(int1)-1))
    println(" ")
    println(" optimal investments (GW) @ 5 years")
    if (b.data.case<=2)
        int2 = " tech.    "
        for n in 1:size(x2)[2]
            int2 *= " "^(7-length("i$n"))*"i$n"
        end
        println(" " * "-" ^ (length(int2)-1))
        println(int2)
        println(" " * "-" ^ (length(int2)-1))
        for p in b.data.ms.P
            println(str_fcn1(x2[p,:],str_tech[p]))
        end
        println(" "*"-"^(length(int2)-1))
    else
        int2 = [" tech.    "," tech.    "," tech.    "]
        for j in 1:3
            for i in 1:Int64(size(x2)[2]/3)
                int2[j] *= " "^(7-length("i$(9*(j-1)+i)"))*"i$(9*(j-1)+i)"
            end
        end
        for j in 1:3
            println(" " * "-" ^ (length(int2[j])-1))
            println(int2[j])
            println(" " * "-" ^ (length(int2[j])-1))
            for p in b.data.ms.P
                println(str_fcn1(x2[p,9*(j-1)+1:9*j],str_tech[p]))
            end
            println(" " * "-" ^ (length(int2[j])-1))
        end
    end
    println(" ")
    println(" */"*"-"^ 68*"/*")
end

function solve_subtree_Benders!(st::Int64,N::Int64)::Tuple{B3_type,S3_type}

    # changing the code to generate B3_type and S3_types from data

    println(" ")
    println("Generating subtree structures...")

    b = gen_B3_type(st,cs,al,w,J,δ)
    s = gen_S3_type(b.data);

    println("")
    println("Stepping off the tree every $N iterations...")
    println("Running the algorithm on the new tree...")
    println("")

    step_0!(b,s)
    while (b.hist.k < b.data.J)
        step!(b,s,N)
        (b.temp.Δ <= b.data.δ) ? break : nothing
    end
    print_summary_c(b,st)
    print_summary_b(b)

    return (b,s)
end

function solve_subtree_Benders!(st::Int64,γ::Float64)::Tuple{B3_type,S3_type}

    # changing the code to generate B3_type and S3_types from data

    println(" ")
    println("Generating subtree structures...")

    b = gen_B3_type(st,cs,al,w,J,δ)
    s = gen_S3_type(b.data);

    println("")
    println("Stepping off the tree after any improvement to δ larger than $(1-γ)...")
    println("Running the algorithm on the new tree...")
    println("")

    step_0!(b,s)
    while (b.hist.k < b.data.J)
        step!(b,s,γ)
        (b.temp.Δ <= b.data.δ) ? break : nothing
    end
    print_summary_c(b,st)
    print_summary_b(b)

    return (b,s)
end

function solve_subtree_Benders!(st::Int64,A::Float64,α::Float64)::Tuple{B3_type,S3_type}

    # changing the code to generate B3_type and S3_types from data

    println(" ")
    println("Generating subtree structures...")

    b = gen_B3_type(st,cs,al,w,J,δ)
    s = gen_S3_type(b.data);

    println("")
    println("Stepping off the tree if local improvements are below $A * exp(-$α*k)...")
    println("Running the algorithm on the new tree...")
    println("")

    step_0!(b,s)
    while (b.hist.k < b.data.J)
        step!(b,s,A,α)
        (b.temp.Δ <= b.data.δ) ? break : nothing
    end
    print_summary_c(b,st)
    print_summary_b(b)

    return (b,s)
end

function solve_subtree_Benders!(st::Int64)::Tuple{B3_type,S3_type}

    # changing the code to generate B3_type and S3_types from data

    println(" ")
    println("Generating subtree structures...")

    b = gen_B3_type(st,cs,al,w,J,δ)
    s = gen_S3_type(b.data);

    println("")
    println("Stepping off the tree if normalised local improvements are below global improvement δ ...")
    println("Running the algorithm on the new tree...")
    println("")

    ω = step2_0!(b,s)[3];
    while (b.hist.k < b.data.J)
        step2!(b,s,ω)
        (b.temp.Δ <= b.data.δ) ? break : nothing
    end
    print_summary_c(b,st)
    print_summary_b(b)

    return (b,s)
end

function step2_0!(b::B3_type,s::S3_type)::Tuple{B3_type,S3_type,Float64}

    s.temp.i = 1
    s.temp.x[:,1] .= round.(vcat(minimum(b.data.mp.xh,dims=2)[:]*.99,minimum(b.data.unc.h,dims=1)[:].*[.99,1.01]); digits=4)
    s.temp.c[:,1] .= round.(minimum(b.data.unc.c,dims=1)[:]*.99; digits=4)
    solv_exact!(s)
    update_s!(s)
    s.temp.x .= vcat(b.data.mp.xh,b.data.unc.h')
    s.temp.c .= b.data.unc.c'
    for s.temp.i in 1:b.data.unc.ni
        run_oracles!(s)
    end
    b.temp.θl .= s.temp.θl
    b.temp.θu .= s.temp.θu
    b.temp.λl .= s.temp.λl
    b.temp.ϕu .= s.temp.ϕu

    ω = findmax(b.data.mp.π[b.data.kidx[w]].*(b.temp.θu[b.data.kidx[w]].-b.temp.θl[b.data.kidx[w]]))[1];

    return (b,s,ω);
end

function step2!(b::B3_type,s::S3_type,ω::Float64)::Tuple{B3_type,S3_type}

    b.hist.k += 1
    b.hist.T[b.hist.k,1] = @elapsed step_a!(b,s)
    b.hist.T[b.hist.k,2] = @elapsed step_b!(b)
    b.hist.T[b.hist.k,3] = @elapsed off_tree = step_c!(b,s,ω)[3]
    b.hist.T[b.hist.k,4] = @elapsed step_d!(b,s)
    b.hist.T[b.hist.k,5] = @elapsed step_e!(b)
    b.hist.T[b.hist.k,6] = @elapsed step_f!(b)

    a1 = "$(round(b.temp.Δ;digits=3))" * "0" ^ (5 - length("$(round(b.temp.Δ%1;digits=3))"))
    a2 = "$(round(sum(b.hist.T[b.hist.k,:]);digits=2))"
    str = " k =" * (" " ^ (4-length("$(b.hist.k)"))) * "$(b.hist.k), "
    str *= "δ =" * (" " ^ (7-length(a1))) * "$(a1) %, "
    str *= "t =" * (" " ^ (8-length(a2))) * "$(a2) s"

    if off_tree
        str *= " (exploring off the tree this iteration)";
    end

    println(str)

    return b,s
end

# Problem with the code below -- shares a type with a different method

#function step_c!(b::B3_type,s::S3_type,ω::Float64)::Tuple{B3_type,S3_type,Bool}
#
#    off_tree = set_Iex2!(b,ω)[2]
#    for s.temp.i in b.temp.Ie
#        solv_exact!(s)
#        update_s!(s)
#    end
#
#    return (b,s,off_tree);
#end

function set_Iex2!(b::B3_type,ω::Float64)::Tuple{B3_type,Bool}

    off_tree = false;

    if length(filter(!iszero,b.data.mp.π)) == 2
        more_π = zeros(756);

        bigger_tree = [5,11,13,14,15,17,23,
                       140,149,158,308,311,314,364,365,366,
                       383,389,391,392,393,395,401,
                       418,419,420,470,473,476,626,635,644];
        for bt in bigger_tree
            more_π[bt] = 1.0;
        end
    elseif length(filter(!iszero,b.data.mp.π)) == 12
        more_π = zeros(756);

        # can only visit nodes where one of the parameters takes mean values
        # these are hard-coded below (based on some Excel manipulations)

        if b.data.mp.π[6] != 0
            # existing tree is for c_ur free
            biggest_tree = [2,4,5,6,8,11,13,14,15,17,20,22,23,24,26,
                            56,59,62,65,68,71,74,77,80,112,113,114,121,122,
                            123,130,131,132,137,139,140,141,143,146,148,149,
                            150,152,155,157,158,159,161,166,167,168,175,176,
                            177,184,185,186,218,221,224,227,230,233,236,239,
                            242,299,302,305,308,311,314,317,320,323,355,356,
                            357,364,365,366,373,374,375,380,382,383,384,386,
                            389,391,392,393,395,398,400,401,402,404,409,410,
                            411,418,419,420,427,428,429,461,464,467,470,473,
                            476,479,482,485,542,545,548,551,554,557,560,563,
                            566,598,599,600,607,608,609,616,617,618,623,625,
                            626,627,629,632,634,635,636,638,641,643,644,645,
                            647,652,653,654,661,662,663,670,671,672,704,707,
                            710,713,716,719,722,725,728];
        elseif b.data.mp.π[12] != 0;
            # existing tree is for c_c02 free
            biggest_tree = [2,5,8,10,11,12,13,14,15,16,17,18,20,23,26,
                            56,59,62,65,68,71,74,77,80,137,140,143,146,149,
                            152,155,158,161,218,221,224,227,230,233,236,239,
                            242,280,281,282,283,284,285,286,287,288,299,302,
                            305,307,308,309,310,311,312,313,314,315,317,320,
                            323,334,335,336,337,338,339,340,341,342,361,362,
                            363,364,365,366,367,368,369,380,383,386,388,389,
                            390,391,392,393,394,395,396,398,401,404,415,416,
                            417,418,419,420,421,422,423,442,443,444,445,446,
                            447,448,449,450,461,464,467,469,470,471,472,473,
                            474,475,476,477,479,482,485,496,497,498,499,500,
                            501,502,503,504,542,545,548,551,554,557,560,563,
                            566,623,626,629,632,635,638,641,644,647,704,707,
                            710,713,716,719,722,725,728];
        else
            # existing tree is for v_CO2 free
            biggest_tree = [4,5,6,10,11,12,13,14,15,16,17,18,22,23,24,
                            112,113,114,121,122,123,130,131,132,139,140,141,
                            148,149,150,157,158,159,166,167,168,175,176,177,
                            184,185,186,280,281,282,283,284,285,286,287,288,
                            307,308,309,310,311,312,313,314,315,334,335,336,
                            337,338,339,340,341,342,355,356,357,361,362,363,
                            364,365,366,367,368,369,373,374,375,382,383,384,
                            388,389,390,391,392,393,394,395,396,400,401,402,
                            409,410,411,415,416,417,418,419,420,421,422,423,
                            427,428,429,442,443,444,445,446,447,448,449,450,
                            469,470,471,472,473,474,475,476,477,496,497,498,
                            499,500,501,502,503,504,598,599,600,607,608,609,
                            616,617,618,625,626,627,634,635,636,643,644,645,
                            652,653,654,661,662,663,670,671,672];
        end

        for bt in biggest_tree
            more_π[bt] = 1.0;
        end
    else
        more_π = ones(756);
    end

    for w in 1:b.data.w

        (val_off,ie_off) = findmax(more_π.*(b.temp.θu[b.data.kidx[w]].-b.temp.θl[b.data.kidx[w]]));
        (val_on,ie_on) = findmax(b.data.mp.π[b.data.kidx[w]].*(b.temp.θu[b.data.kidx[w]].-b.temp.θl[b.data.kidx[w]]));

        #if (b.temp.Δ > 0) && ((val_on/ω) < (b.temp.Δ*exp(-0.5*b.hist.k)))
        if (b.temp.Δ > 0) && ((val_on/ω) < (b.temp.Δ/(b.hist.k)^2))
            ie = ie_off;
            off_tree = true;
        else
            ie = ie_on;
        end

        b.temp.Ie[w] = b.data.iidx[w][ie];

        str = "Solving the subproblem at node $ie";
        str *= " "^(4-length("$ie"));
        print(str)
    end

    return (b,off_tree)
end

function tree_1()::Array{Float64,1}
    # tree 1 corresponds to an uncertain CO2 budget
    π1 = zeros(757);
    π1[1] = 1.0;

    π1[14:16] = (1/3)*ones(3);

    π1[365:367] = (1/9)*ones(3);
    π1[392:394] = (1/9)*ones(3);
    π1[419:421] = (1/9)*ones(3);

    return π1
end

function tree_2()::Array{Float64,1}
    # tree 2 corresponds to an uncertain CO2 cost
    π1 = zeros(757);
    π1[1] = 1.0;

    π1[12] = 1/3;
    π1[15] = 1/3;
    π1[18] = 1/3;

    π1[309] = 1/9;
    π1[312] = 1/9;
    π1[315] = 1/9;

    π1[390] = 1/9;
    π1[393] = 1/9;
    π1[396] = 1/9;

    π1[471] = 1/9;
    π1[474] = 1/9;
    π1[477] = 1/9;

    return π1
end

function tree_3()::Array{Float64,1}
    # tree 3 corresponds to an uncertain uranium fuel cost
    π1 = zeros(757);
    π1[1] = 1.0;

    π1[6] = 1/3;
    π1[15] = 1/3;
    π1[24] = 1/3;

    π1[141] = 1/9;
    π1[150] = 1/9;
    π1[159] = 1/9;

    π1[384] = 1/9;
    π1[393] = 1/9;
    π1[402] = 1/9;

    π1[627] = 1/9;
    π1[636] = 1/9;
    π1[645] = 1/9;

    return π1
end

function tree_12(hs::Array{Float64,2},cs::Array{Float64,2})::Array{Float64,1}
    # tree 12 corresponds to uncertain CO2 budget and CO2 cost
    π1 = zeros(757);
    π1[1] = 1.0;

    for i in 1:756
        if abs(cs[i,2] - 10) < 0.0001
            if abs(hs[i,2] + 1.1) < 0.0001
                π1[i+1] = (1/9);    # this is a first stage node
            else
                π1[i+1] = (1/81);   # this is a second stage node
            end
        end
    end

    return π1
end

function tree_13(hs::Array{Float64,2},cs::Array{Float64,2})::Array{Float64,1}
    # tree 12 corresponds to uncertain CO2 budget and CO2 cost
    π1 = zeros(757);
    π1[1] = 1.0;

    for i in 1:756
        if abs(cs[i,1] - 60) < 0.0001
            if abs(hs[i,2] + 1.1) < 0.0001
                π1[i+1] = (1/9);    # this is a first stage node
            else
                π1[i+1] = (1/81);   # this is a second stage node
            end
        end
    end

    return π1
end

function tree_23(hs::Array{Float64,2},cs::Array{Float64,2})::Array{Float64,1}
    # tree 12 corresponds to uncertain CO2 budget and CO2 cost
    π1 = zeros(757);
    π1[1] = 1.0;

    for i in 1:756
        if abs(hs[i,1] - 0.8) < 0.0001 && abs(hs[i,2] + 1.1) < 0.001
            π1[i+1] = (1/9);    # this is a first stage node
        elseif abs(hs[i,1] - 0.6) < 0.0001 && abs(hs[i,2] + 1.25) < 0.001
            π1[i+1] = (1/81);   # this is a second stage node
        end
    end

    return π1
end

function expand_subtree(b::B3_type,cs::Int64)::B3_type
    # update the probabilities and copy the x0 values into the right places
    if cs == 1
        π1 = tree_1();
    elseif cs == 2
        π1 = tree_2();
    elseif cs == 3
        π1 = tree_3();
    elseif cs == 12
        π1 = tree_12(b.data.unc.h,b.data.unc.c);
    elseif cs == 13
        π1 = tree_13(b.data.unc.h,b.data.unc.c);
    elseif cs == 23
        π1 = tree_23(b.data.unc.h,b.data.unc.c);
    end

    b.data.mp.π0 = π1[1:28];
    b.data.mp.π = π1[2:757];

    c01 = @constraint(b.rmp.m, b.rmp.m[:f] >= exp10(-6)*sum(b.data.mp.π0[i0]*sum(b.data.mp.ci[p,i0]*b.rmp.m[:x0][p,i0] for p in b.data.ms.P) for i0 in b.data.ms.I0) + exp10(-6)*b.data.mp.κ*sum(b.data.mp.π[i]*sum(b.data.mp.cf[p]*b.rmp.m[:x][p,i] for p in b.data.ms.P) for i in b.data.ms.I) );
    @objective(b.rmp.m, Min, b.rmp.m[:f] + b.data.mp.κ*sum(b.data.mp.π[i]*b.rmp.m[:β][i] for i in b.data.ms.I) );

    optimize!(b.rmp.m);
    return b
end

function expand_subtree(b::B3_type,cs::Int64,nodeA::Array{Float64,1},nodeB::Array{Float64,1})::B3_type
    # update the probabilities and copy the x0 values into the right places
    if cs == 1
        π1 = tree_1();
        for k in 1:12
            JuMP.fix(b.rmp.m[:x0][k,1], nodeA[k], force=true);
            JuMP.fix(b.rmp.m[:x0][k,14], nodeB[k], force=true);
            JuMP.fix(b.rmp.m[:x0][k,15], nodeB[k], force=true);
            JuMP.fix(b.rmp.m[:x0][k,16], nodeB[k], force=true);
        end
    elseif cs == 2
        π1 = tree_2();
        for k in 1:12
            JuMP.fix(b.rmp.m[:x0][k,1], nodeA[k], force=true);
            JuMP.fix(b.rmp.m[:x0][k,12], nodeB[k], force=true);
            JuMP.fix(b.rmp.m[:x0][k,15], nodeB[k], force=true);
            JuMP.fix(b.rmp.m[:x0][k,18], nodeB[k], force=true);
        end
    elseif cs == 3
        π1 = tree_3();
        for k in 1:12
            JuMP.fix(b.rmp.m[:x0][k,1], nodeA[k], force=true);
            JuMP.fix(b.rmp.m[:x0][k,6], nodeB[k], force=true);
            JuMP.fix(b.rmp.m[:x0][k,15], nodeB[k], force=true);
            JuMP.fix(b.rmp.m[:x0][k,24], nodeB[k], force=true);
        end
    elseif cs == 12
        π1 = tree_12(b.data.unc.h,b.data.unc.c);
    elseif cs == 13
        π1 = tree_13(b.data.unc.h,b.data.unc.c);
    elseif cs == 23
        π1 = tree_23(b.data.unc.h,b.data.unc.c);
    end

    b.data.mp.π0 = π1[1:28];
    b.data.mp.π = π1[2:757];

    c01 = @constraint(b.rmp.m, b.rmp.m[:f] >= exp10(-6)*sum(b.data.mp.π0[i0]*sum(b.data.mp.ci[p,i0]*b.rmp.m[:x0][p,i0] for p in b.data.ms.P) for i0 in b.data.ms.I0) + exp10(-6)*b.data.mp.κ*sum(b.data.mp.π[i]*sum(b.data.mp.cf[p]*b.rmp.m[:x][p,i] for p in b.data.ms.P) for i in b.data.ms.I) );
    @objective(b.rmp.m, Min, b.rmp.m[:f] + b.data.mp.κ*sum(b.data.mp.π[i]*b.rmp.m[:β][i] for i in b.data.ms.I) );

    optimize!(b.rmp.m);
    return b
end


function check_bounds(b::B3_type,s::S3_type)
    # compare upper and lower bounds without iterating

    L = exp10(6)*objective_value(b.rmp.m);
    print("L = $L, ")

    # b.temp.β .= value.(b.rmp.m[:β])
    b.temp.x .= value.(b.rmp.m[:x])
    s.temp.x .= b.temp.x

    update_s!(s)

    for s.temp.i in 1:b.data.unc.ni
        run_oracles!(s)
    end

    b.temp.θl .= s.temp.θl;
    b.temp.θu .= s.temp.θu;
    b.temp.λl .= s.temp.λl;
    b.temp.ϕu .= s.temp.ϕu;

    U = exp10(6)*( value(b.rmp.m[:f])+(b.data.mp.κ*b.data.mp.π'*b.temp.θu) );
    # U = b.hist.U[b.hist.k]; # this is the old U, ie expanding bounds from TO
    print("U = $U, ")

    println("Δ = $(U-L)")

end
