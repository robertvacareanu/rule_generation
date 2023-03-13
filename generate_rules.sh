#!/bin/bash

# We created multiple files for different rules simply to keep things separate
# The disadvantage to this is that they share quite a lot in common, so if we modify something
# We have to make sure we don't miss it in the other files


/home/rvacareanu/miniconda3/envs/softrules/bin/python main2_softrules_fstacred.py --out-dir data/test_rules/ --min-span-length 1 --max-span-length 15 --num-queries 2 --num-matches 3 --rule_type 'surface' --save_path "/home/rvacareanu/projects/temp/odinsynth2/data/softrules/fstacred/surface.jsonl"
/home/rvacareanu/miniconda3/envs/softrules/bin/python main2_softrules_fstacred.py --out-dir data/test_rules/ --min-span-length 1 --max-span-length 15 --num-queries 2 --num-matches 3 --rule_type 'simplified_syntax' --save_path "/home/rvacareanu/projects/temp/odinsynth2/data/softrules/fstacred/simplified_syntax.jsonl"
/home/rvacareanu/miniconda3/envs/softrules/bin/python main2_softrules_fstacred.py --out-dir data/test_rules/ --min-span-length 1 --max-span-length 15 --num-queries 2 --num-matches 3 --rule_type 'syntax' --save_path "/home/rvacareanu/projects/temp/odinsynth2/data/softrules/fstacred/syntax.jsonl"
/home/rvacareanu/miniconda3/envs/softrules/bin/python main2_softrules_fstacred.py --out-dir data/test_rules/ --min-span-length 1 --max-span-length 15 --num-queries 2 --num-matches 3 --rule_type 'enhanced_syntax' --save_path "/home/rvacareanu/projects/temp/odinsynth2/data/softrules/fstacred/enhanced_syntax.jsonl"

/home/rvacareanu/miniconda3/envs/softrules/bin/python main2_softrules_tacred.py --out-dir data/test_rules/ --min-span-length 1 --max-span-length 15 --num-queries 2 --num-matches 3 --rule_type 'surface' --save_path "/home/rvacareanu/projects/temp/odinsynth2/data/softrules/tacred/surface.jsonl"
/home/rvacareanu/miniconda3/envs/softrules/bin/python main2_softrules_tacred.py --out-dir data/test_rules/ --min-span-length 1 --max-span-length 15 --num-queries 2 --num-matches 3 --rule_type 'simplified_syntax' --save_path "/home/rvacareanu/projects/temp/odinsynth2/data/softrules/tacred/simplified_syntax.jsonl"
/home/rvacareanu/miniconda3/envs/softrules/bin/python main2_softrules_tacred.py --out-dir data/test_rules/ --min-span-length 1 --max-span-length 15 --num-queries 2 --num-matches 3 --rule_type 'syntax' --save_path "/home/rvacareanu/projects/temp/odinsynth2/data/softrules/tacred/syntax.jsonl"
/home/rvacareanu/miniconda3/envs/softrules/bin/python main2_softrules_tacred.py --out-dir data/test_rules/ --min-span-length 1 --max-span-length 15 --num-queries 2 --num-matches 3 --rule_type 'enhanced_syntax' --save_path "/home/rvacareanu/projects/temp/odinsynth2/data/softrules/tacred/enhanced_syntax.jsonl"

/home/rvacareanu/miniconda3/envs/softrules/bin/python main2_softrules.py --out-dir data/test_rules/ --min-span-length 1 --max-span-length 15 --num-queries 2 --num-matches 3 --rule_type 'syntax' --save_path "/home/rvacareanu/projects/temp/odinsynth2/data/softrules/syntax/rules.jsonl"
/home/rvacareanu/miniconda3/envs/softrules/bin/python main2_softrules.py --out-dir data/test_rules/ --min-span-length 1 --max-span-length 15 --num-queries 2 --num-matches 3 --rule_type 'simplified_syntax' --save_path "/home/rvacareanu/projects/temp/odinsynth2/data/softrules/simplified_syntax/rules.jsonl"
/home/rvacareanu/miniconda3/envs/softrules/bin/python main2_softrules.py --out-dir data/test_rules/ --min-span-length 1 --max-span-length 15 --num-queries 2 --num-matches 3 --rule_type 'surface' --save_path "/home/rvacareanu/projects/temp/odinsynth2/data/softrules/surface/rules.jsonl"
/home/rvacareanu/miniconda3/envs/softrules/bin/python main2_softrules.py --out-dir data/test_rules/ --min-span-length 1 --max-span-length 15 --num-queries 2 --num-matches 3 --rule_type 'enhanced_syntax' --save_path "/home/rvacareanu/projects/temp/odinsynth2/data/softrules/enhanced_syntax/rules.jsonl"



