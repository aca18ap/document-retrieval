python IR_engine.py -o test.txt -w tf -p -s
python eval_ir.py cacm_gold_std.txt test.txt
python IR_engine.py -o test.txt -w tf -p 
python eval_ir.py cacm_gold_std.txt test.txt
python IR_engine.py -o test.txt -w tf -s
python eval_ir.py cacm_gold_std.txt test.txt
python IR_engine.py -o test.txt -w tf 
python eval_ir.py cacm_gold_std.txt test.txt

python IR_engine.py -o test.txt -w tfidf -p -s
python eval_ir.py cacm_gold_std.txt test.txt
python IR_engine.py -o test.txt -w tfidf -p 
python eval_ir.py cacm_gold_std.txt test.txt
python IR_engine.py -o test.txt -w tfidf -s
python eval_ir.py cacm_gold_std.txt test.txt
python IR_engine.py -o test.txt -w tfidf 
python eval_ir.py cacm_gold_std.txt test.txt

python IR_engine.py -o test.txt -w binary -p -s
python eval_ir.py cacm_gold_std.txt test.txt
python IR_engine.py -o test.txt -w binary -p 
python eval_ir.py cacm_gold_std.txt test.txt
python IR_engine.py -o test.txt -w binary -s
python eval_ir.py cacm_gold_std.txt test.txt
python IR_engine.py -o test.txt -w binary 
python eval_ir.py cacm_gold_std.txt test.txt


pause