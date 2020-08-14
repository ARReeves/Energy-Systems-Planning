#===============================================================

Code to model scenario trees, including functions for splitting
and reducing scenarios at specified nodes.

===============================================================#

using Gadfly
using GraphPlot
using LightGraphs

const EMPTY_SET = Set{Int64}();

mutable struct Node
    label::Int64
    sources::Set{Int64}
    targets::Set{Int64}
end

isroot = function(n::Node)
    return isempty(n.sources)
end

isleaf = function(n::Node)
    return isempty(n.targets)
end

const Quiver = Dict{Int64,Node};

function Qplot(q::Quiver)
    # plot a quiver as a simple directed graph
    g = DiGraph();
    nodes = sort(collect(keys(q)));

    for n in nodes
        add_vertex!(g);
        for s in q[n].sources
            x = findlast(λ -> λ == q[s].label,nodes);
            y = findlast(λ -> λ == q[n].label,nodes)
            add_edge!(g,x,y);
        end
    end
    gplot(g,nodelabel=nodes)
end

mutable struct ScenarioTree
    # for our purposes a scenario tree is simply a quiver
    # in which every node n is assigned a stage s(n), a
    # probability p(n) of being visited and a range (ie upper
    # and lower bounds) for the associated random variable ξ_s(n)
    tree::Quiver
    p::Dict{Int64,Float64}
    stages::Dict{Int64,Int64}
    upper::Dict{Int64,Float64}
    lower::Dict{Int64,Float64}
end

function Qplot(t::ScenarioTree)
    Qplot(t.tree)
end

function link!(n1::Node, n2::Node)
    # connect two nodes, changing them in place
    push!(n1.targets,n2.label);
    push!(n2.sources,n1.label);
end

function split!(n1::Node,n2::Node)
    # disconnect two nodes, changing them in place
    delete!(n1.targets,n2.label);
    delete!(n2.sources,n1.label);
end

function trim!(q::Quiver)
    # remove all arrows that point to nodes not in the
    # given quiver
    nodes = values(q);
    for n in nodes
        filter!(λ -> (λ in nodes), n.sources);
        filter!(λ -> (λ in nodes), n.targets);
    end
end

function descendants(q::Quiver, n::Node)
    # return a set of all the labels of descendants of
    # node n in quiver Q
    v = n.targets;
    if !isempty(v)
        for x in v
            v = union(v,descendants(q,q[x]));
        end
    end
    return v
end

function split!(q::Quiver, n::Node)
    # splits a quiver at a given node
    if isroot(n)
        println("WARNING: Splitting on a root node.")
    end

    old_labels = sort(collect(keys(q)));
    to_copy = sort(collect(descendants(q,n)));

    d = maximum(old_labels)+1-n.label;
    i = maximum(old_labels)+1;
    new_targets = Set(map(λ -> (λ+d),collect(n.targets)));
    new_node = Node(i,n.sources,new_targets);

    for v in n.sources
        link!(q[v],new_node);
    end

    new_q = Quiver(i => new_node);

    while !isempty(to_copy)
        x = popfirst!(to_copy);
        i = i+1;

        new_sources = Set(map(λ -> (λ+d),
                              collect(q[x].sources)));
        new_targets = Set(map(λ -> (λ+d),
                              collect(q[x].targets)));
        new_node = Node(i,new_sources,new_targets);
        merge!(new_q,Quiver(i => new_node));
    end

    merge!(q,new_q);
end

function split!(t::ScenarioTree, n::Node, v::Float64)
    # splits a scenario tree at a given node at a given
    # critical value

    if isroot(n)
        println("WARNING: Splitting on a root node.")
    end

    old_labels = sort(collect(keys(t.tree)));
    to_copy = sort(collect(descendants(t.tree,n)));

    d = maximum(old_labels)+1-n.label;
    i = maximum(old_labels)+1;
    new_targets = Set(map(λ -> (λ+d),collect(n.targets)));
    new_node = Node(i,n.sources,new_targets);

    prob = (v-t.lower[n.label])/(t.upper[n.label] - t.lower[n.label]);

    t.stages[i] = t.stages[n.label];
    t.upper[i] = t.upper[n.label];
    t.upper[n.label] = v;
    t.lower[i] = v;

    t.p[i] = (1-prob) * t.p[n.label];
    t.p[n.label] = prob * t.p[n.label];

    for v in n.sources
        link!(t.tree[v],new_node);
    end

    new_q = Quiver(i => new_node);

    while !isempty(to_copy)
        x = popfirst!(to_copy);
        i = i+1;


        t.stages[i] = t.stages[x];
        t.upper[i] = t.upper[x];
        t.lower[i] = t.lower[x];

        new_sources = Set(map(λ -> (λ+d),
                              collect(t.tree[x].sources)));
        new_targets = Set(map(λ -> (λ+d),
                              collect(t.tree[x].targets)));
        new_node = Node(i,new_sources,new_targets);

        t.p[i] = (1-prob) * t.p[x];
        t.p[x] = prob * t.p[x];

        merge!(new_q,Quiver(i => new_node));
    end

    merge!(t.tree,new_q);
    return t
end
