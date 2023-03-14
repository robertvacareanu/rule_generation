# Rule Generation
Generating random rules.
The code is slightly adapted from [odinsynth2](https://github.com/marcovzla/odinsynth2).

The code is organized as follows:
- Has 3 "main" files: `main2_softrules_fstacred`, `main2_softrules_tacred`, and `main2_softrules`.
    - `main2_softrules_fstacred` -> Generates rules for the few-shot TACRED dataset
    - `main2_softrules_tacred` -> Generates rules for the TACRED dataset
    - `main2_softrules` -> Generates rules from UMBC
- A new `rulegen.py` file called `rulegen2.py`. This file contains, additionally, a couple of additional functions. They were added in order to generate the following types of rules (we exemplify the rule using the sentence: `The quick brown fox jumped over the lazy dog .`):
    - `surface`           -> A rule that uses only surface constraints (e.g. `[tag=JJ] [tag=JJ] [word=fox]` to match `quick brown fox`)
    - `simplified syntax` -> A rule that uses only surface constraints, but over the dependency graph (e.g. `[word=fox] [word=jumped] [tag=NN]` to match the words `fox jumped dod` (or the two animals, if namdd captures are added))
    - `syntax`            -> A rule that uses only syntax constraints (e.g. `[word=fox] <nsubj >dobj [word=dog]`)
    - `ehanced syntax`    -> A rule that uses a mix of surface and syntax constraints (e.g. `[word=fox] <nsubj jumped >dobj [word=dog]`)
