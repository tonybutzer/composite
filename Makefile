gitbig:
	find . -size +10M
	echo du -a ./ | sort -n -r | head -n 20
	#for file in `find . -size +10M`; do ls -lh $$file; done

publish:
	git config --global user.email tonybutzer@gmail.com
	git config --global user.name tonybutzer
	git config --global push.default simple
	git add .
	git commit -m "automatic git update from Makefile"
	git push


lite:
	(cd opt; git clone http://github.com/tonybutzer/lite-stac)
