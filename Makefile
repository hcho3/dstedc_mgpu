all: prof
.PHONY: prof

prof:
	$(MAKE) -C prof/

clean:
	$(MAKE) -C prof/ clean
