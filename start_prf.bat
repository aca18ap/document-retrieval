python IR_engine.py -o test.txt -w tf -p -s -f
python eval_ir.py cacm_gold_std.txt test.txt
python IR_engine.py -o test.txt -w tf -p  -f
python eval_ir.py cacm_gold_std.txt test.txt
python IR_engine.py -o test.txt -w tf -s -f
python eval_ir.py cacm_gold_std.txt test.txt
python IR_engine.py -o test.txt -w tf  -f
python eval_ir.py cacm_gold_std.txt test.txt

python IR_engine.py -o test.txt -w tfidf -p -s -f
python eval_ir.py cacm_gold_std.txt test.txt
python IR_engine.py -o test.txt -w tfidf -p  -f
python eval_ir.py cacm_gold_std.txt test.txt
python IR_engine.py -o test.txt -w tfidf -s -f
python eval_ir.py cacm_gold_std.txt test.txt
python IR_engine.py -o test.txt -w tfidf  -f
python eval_ir.py cacm_gold_std.txt test.txt





pause