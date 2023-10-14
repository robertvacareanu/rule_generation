import copy
import random
from typing import Optional, List, Tuple
import networkx as nx
from odinson.gateway import *
from odinson.ruleutils import *
from odinson.ruleutils.queryast import *
from odinson.ruleutils.queryparser import parse_traversal
from index import IndexedCorpus
from util import weighted_choice, random_span, random_spans
from odinson.gateway import *
from odinson.ruleutils import *
from odinson.ruleutils.queryast import *
from odinson.ruleutils.queryparser import parse_traversal
from typing import Optional, List

class RuleGeneration:
    def __init__(
        self, 
        corpus: IndexedCorpus, 
        min_span_length: int = 1,
        max_span_length: int = 5,
        num_matches: int = 10,
        fields: Optional[dict] = None,
        constraint_actions: Optional[dict] = None,
        surface_actions: Optional[dict] = None,
        quantifiers: Optional[dict] = None,
    ):
        self.corpus = corpus

        self.min_span_length = min_span_length
        self.max_span_length = max_span_length
        self.num_matches = num_matches

        if fields:
            self.fields = fields
        else:
            self.fields = {
                "lemma": 5,
                "word": 1,
                "tag": 1,
            }

        if constraint_actions:
            self.constraint_actions = constraint_actions
        else:
            self.constraint_actions = {
                "or": 2,
                "and": 0,
                "not": 1,
                "stop": 5,
            }

        if surface_actions:
            self.surface_actions = surface_actions
        else:
            self.surface_actions = {
                "or": 1,
                "concat": 10,
                "quantifier": 2,
                "stop": 5,
            }

        if quantifiers:
            self.quantifiers = quantifiers
        else:
            self.quantifiers = {
                "?": 5,
                "*": 1,
                "+": 4,
            }

    def random_surface_rule(
        self,
        sentence: Sentence,
        span: tuple[int, int],
    ) -> Tuple[str, str]:
        """
        Returns a random surface rule.
        If sentence and span are provided, the generated rule will match it.
        If only sentence is provided, the generated rule will match a random span in the sentence.
        If a document is provided, a random sentence and span will be used to generate the rule.
        If sentence is provided then document is ignored.
        If sentence is not provided then span is ignored.
        """
        # print("\tRANDOM_SURFACE_RULE open")
        # ensure we have a sentence and a span
        start, stop = span
        if start == stop:
            return None, ''

        # make a token constraint for each token in the span
        constraints = self.make_field_constraints(sentence, start, stop)
        # add some random token constraints
        constraints = self.add_random_constraints(constraints)
        # wrap constraints
        nodes = [TokenSurface(c) for c in constraints]
        all_nodes = self.add_random_surface(nodes)

        sentence_tokens = sentence.get_field("raw").tokens
        matched_tokens = [sentence_tokens[x] for x in range(start, stop)]


        # We also return the matched tokens. This is needed for the way we will use this in the future
        return str([self.concat_surface_nodes(x) for x in all_nodes][-1]), ' '.join(matched_tokens)

    def make_field_constraints(
        self, sentence: Sentence, start: int, stop: int
    ) -> list[Constraint]:
        """
        Gets a sentence and the indices of a span within the sentence.
        Returns a list of token constraints, one for each token in the span.
        """
        constraints = []
        for i in range(start, stop):
            name = weighted_choice(self.fields)
            value = sentence.get_field(name).tokens[i]
            while (name == 'entity' and value == 'O'):
                name = weighted_choice(self.fields)
                value = sentence.get_field(name).tokens[i]

            c = FieldConstraint(ExactMatcher(name), ExactMatcher(value))
            constraints.append(c)
        return constraints

    def add_random_constraints(self, constraints: list[Constraint]) -> List[Constraint]:
        """
        Gets a list of token constraints and a number of modifications to perform.
        Returns a new list of token constraints with the same length as the original.
        """
        while True:
            cs = copy.copy(constraints)
            i = random.randrange(len(cs))
            action = weighted_choice(self.constraint_actions)
            if action == "stop":
                break
            elif action == "or":
                # make pattern
                lookbehind = self.concat_surface_nodes(self.wrap_constraints(cs[:i])) if i > 0 else None
                lookahead = self.concat_surface_nodes(self.wrap_constraints(cs[i+1:])) if i < len(cs) - 1 else None
                pattern = ""
                if lookbehind:
                    pattern += f"(?<={lookbehind}) "
                pattern += "[]"
                if lookahead:
                    pattern += f" (?={lookahead})"
                # execute modified rule
                results = self.corpus.search(pattern, max_hits=self.num_matches)
                # find an alternative
                score_doc = random.choice(results.docs)
                sentence = self.corpus.get_sentence(score_doc)
                f = weighted_choice(self.fields)
                v = sentence.get_field(f).tokens[score_doc.matches[0].start]
                # add new constraint
                new_constraint = FieldConstraint(ExactMatcher(f), ExactMatcher(v))
                if random.random() < 0.5:
                    cs[i] = OrConstraint(cs[i], new_constraint)
                else:
                    cs[i] = OrConstraint(new_constraint, cs[i])
            elif action == "and":
                # TODO
                continue
            elif action == "not":
                # avoid double negation
                if isinstance(cs[i], NotConstraint):
                    continue
                cs[i] = NotConstraint(cs[i])
            # if self.check_constraint_modification(constraints, cs):
            constraints = cs
        return constraints

    def add_random_surface(self, nodes: list[Surface]) -> list[Surface]:
        """
        Gets a list of surface nodes and a number of modifications to perform.
        Returns a new list of surface nodes with length less than or equal to the original.
        """
        resulting_surfaces = [nodes]
        while True:
            ns = copy.copy(nodes)
            action = weighted_choice(self.surface_actions)
            if action == "stop":
                break
            elif action == "concat":
                # we need at least two nodes to concatenate
                if len(ns) < 2:
                    continue
                # choose random node (can't be the last one)
                i = random.randrange(len(ns) - 1)
                # concatenate selected node and the next one,
                # and replace concatenated nodes with the new concatenation
                ns[i : i + 2] = [ConcatSurface(ns[i], ns[i + 1])]
                # we won't count this as a modification
                continue
            elif action == "or":
                if len(ns) == 1:
                    # if we only have one node then making an OR is easy:
                    # 1) make a random surface rule
                    surf = self.random_surface_rule()[-1]
                    # 2) make or node with our current node and our new surface rule
                    if random.random() < 0.5:
                        ns[0] = OrSurface(ns[0], surf)
                    else:
                        ns[0] = OrSurface(surf, ns[0])
                else:
                    # choose random node
                    i = random.randrange(len(ns))
                    repl = RepeatSurface(WildcardSurface(), 1, self.max_span_length)
                    # nodes than aren't involved in the OR should still match
                    lookbehind = self.concat_surface_nodes(ns[:i]) if i > 0 else None
                    lookahead = self.concat_surface_nodes(ns[i+1:]) if i < len(ns) - 1 else None
                    # construct pattern
                    pattern = ""
                    if lookbehind:
                        pattern += f"(?<={lookbehind}) "
                    pattern += str(repl)
                    if lookahead:
                        pattern += f" (?={lookahead})"
                    # perform search
                    results = self.corpus.search(pattern, self.num_matches)
                    if results.total_hits == 0:
                        continue
                    score_doc = random.choice(results.docs)
                    sentence = self.corpus.get_sentence(score_doc)
                    span = (score_doc.matches[0].start, score_doc.matches[0].end)
                    # find an alternative clause
                    surf = self.random_surface_rule(sentence=sentence, span=span)[-1]
                    # make OR node
                    if random.random() < 0.5:
                        ns[i] = OrSurface(ns[i], surf)
                    else:
                        ns[i] = OrSurface(surf, ns[i])
            elif action == "quantifier":
                # choose random node
                i = random.randrange(len(ns))
                # don't repeat repetitions
                if isinstance(ns[i], RepeatSurface):
                    continue
                # choose random quantifier
                quantifier = weighted_choice(self.quantifiers)
                # wrap selected node with quantifier
                if quantifier == "?":
                    ns[i] = RepeatSurface(ns[i], 0, 1)
                elif quantifier == "*":
                    ns[i] = RepeatSurface(ns[i], 0, None)
                elif quantifier == "+":
                    ns[i] = RepeatSurface(ns[i], 1, None)
            # # confirm that new rule is valid
            # if self.check_surface_modification(nodes, ns):
            #     resulting_surfaces.append(ns)
            #     nodes = ns
        # return surface nodes
        return resulting_surfaces

    def concat_surface_nodes(self, nodes: list[Surface]) -> Surface:
        """
        Gets a list of surface nodes and returns a single surface node
        with their concatenation.
        """
        rule = nodes[0]
        for n in nodes[1:]:
            rule = ConcatSurface(rule, n)
        return rule

    def wrap_constraints(self, constraints: list[Constraint]) -> list[Surface]:
        new_constraints = []
        for c in constraints:
            nc = WildcardSurface() if isinstance(c, WildcardConstraint) else TokenSurface(c)
            new_constraints.append(nc)
        return new_constraints

    def check_constraint_modification(self, old_constraints: list[Constraint], new_constraints: list[Constraint]) -> bool:
        """
        Checks that the results of the new_constraints are non-empty and different
        than the results of old_constraints.
        """
        old_nodes = self.wrap_constraints(old_constraints)
        new_nodes = self.wrap_constraints(new_constraints)
        return self.check_surface_modification(old_nodes, new_nodes)

    def check_surface_modification(self, old_nodes: list[Surface], new_nodes: list[Surface]) -> bool:
        """
        Checks that the results of the new_nodes are non-empty and different
        than the results of old_nodes.
        """
        new_rule = self.concat_surface_nodes(new_nodes)
        new_results = self.corpus.search(new_rule, 1)
        if new_results.total_hits == 0:
            return False
        old_rule = self.concat_surface_nodes(old_nodes)
        old_results = self.corpus.search(old_rule, 1)
        return new_results.total_hits != old_results.total_hits

    def random_enhanced_traversal_rule(
        self,
        sentence: Sentence,
        span: tuple[List[int], List[int]],
    ) -> Tuple[str, str]:
        # ensure we have a sentence
        # get two random spans for the source and the target
        [source, target] = span
        # find candidate paths from source to target
        dependencies = sentence.get_field('dependencies')
        digraph = dependencies.to_networkx()
        all_shortest_paths = nx.shortest_path(digraph)
        candidate_paths = []
        # print("#######################")
        for s in range(*source):
            for t in range(*target):
                try:
                    p = all_shortest_paths[s][t]
                    # print(s, t, p)
                    if len(p) > 0:
                        candidate_paths.append(p)
                except:
                    pass
        path = min(candidate_paths, key=lambda x: len(x))
        steps = [digraph.edges[e[0], e[1]]['label'] for e in nx.path_graph(path).edges]
        sentence_tokens = sentence.get_field("raw").tokens
        matched_tokens = [sentence_tokens[x] for x in path][1:-1]
        # print(matched_tokens)
        constraints = []
        for i in path[1:-1]:
            name = weighted_choice(self.fields)
            value = sentence.get_field(name).tokens[i]
            c = FieldConstraint(ExactMatcher(name), ExactMatcher(value))
            constraints.append(c)

        nodes = [TokenSurface(c) for c in constraints]

        new_nodes = [steps[0]]
        for (c, s) in zip(matched_tokens, steps[1:]):
            new_nodes.append(str(c))
            new_nodes.append(str(s))

        # Return the syntax constraints concatenated
        # We also return the matched tokens. This is needed for the way we will use this in the future
        return ' '.join(new_nodes), ' '.join(matched_tokens)

    def random_traversal_rule(
        self,
        sentence: Sentence,
        span: tuple[List[int], List[int]],
    ) -> Tuple[str, str]:
        # ensure we have a sentence
        # get two random spans for the source and the target
        [source, target] = span
        # find candidate paths from source to target
        dependencies = sentence.get_field('dependencies')
        digraph = dependencies.to_networkx()
        all_shortest_paths = nx.shortest_path(digraph)
        candidate_paths = []
        # print("#######################")
        for s in range(*source):
            for t in range(*target):
                try:
                    p = all_shortest_paths[s][t]
                    # print(s, t, p)
                    if len(p) > 0:
                        candidate_paths.append(p)
                except:
                    pass


        path = min(candidate_paths, key=lambda x: len(x))
        steps = [digraph.edges[e[0], e[1]]['label'] for e in nx.path_graph(path).edges]

        sentence_tokens = sentence.get_field("raw").tokens
        # We skip first and last because first will be part of the first entity and last will be part of the last entity
        matched_tokens = [sentence_tokens[x] for x in path][1:-1]

        # Return the syntax constraints concatenated
        # We also return the matched tokens. This is needed for the way we will use this in the future
        return ' '.join(steps), ' '.join(matched_tokens)

    def random_simplified_traversal_rule(
        self,
        sentence: Sentence,
        span: tuple[List[int], List[int]],
    ) -> Tuple[List[str], str]:
        # ensure we have a sentence
        # get two random spans for the source and the target
        [source, target] = span
        # find candidate paths from source to target
        dependencies = sentence.get_field('dependencies')
        digraph = dependencies.to_networkx()
        all_shortest_paths = nx.shortest_path(digraph)
        candidate_paths = []
        # print("#######################")
        for s in range(*source):
            for t in range(*target):
                try:
                    p = all_shortest_paths[s][t]
                    # print(s, t, p)
                    if len(p) > 0:
                        candidate_paths.append(p)
                except:
                    pass
        # print("#######################")
        # print(sentence)
        # choose a path
        # TODO maybe pick the shortest instead?
        path = min(candidate_paths, key=lambda x: len(x))
        steps = [digraph.edges[e[0], e[1]]['label'] for e in nx.path_graph(path).edges]
        sentence_tokens = sentence.get_field("raw").tokens
        matched_tokens = [sentence_tokens[x] for x in path][1:-1]
        # print(matched_tokens)
        constraints = []
        for i in path[1:-1]:
            name = weighted_choice(self.fields)
            value = sentence.get_field(name).tokens[i]
            c = FieldConstraint(ExactMatcher(name), ExactMatcher(value))
            constraints.append(c)

        nodes = [TokenSurface(c) for c in constraints]
        # print([str(x) for x in nodes])
        # print(' '.join(matched_tokens))
        # We return the constraints as a list, not as a full rule. This is because we want to make them with (<<|>>) to be a valid rule
        # We also return the matched tokens. This is needed for the way we will use this in the future
        return [str(x) for x in nodes], ' '.join(matched_tokens)


