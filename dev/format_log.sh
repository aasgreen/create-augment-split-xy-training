#!\bin\bash
#-r is advanced regex
#-n is print only matching lines
#/p flag is somehow related to print
#using search and replace. Match the whole line, and replace with groups 1 and 2
sed -r -n "s/.*Training loss\: ([0-9]*\.[0-9]*).*Validation loss\: ([0-9]\.[0-9]*).*/\1 \2/p" log.dat > loss.dat
