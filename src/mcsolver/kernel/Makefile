CC := gcc
CFLAGS := -O3 -fopenmp -static -static-libgcc
LDFLAGS := -shared

TARGET := monte_carlo_solver.dll
SRC := monte_carlo_solver.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(LDFLAGS) -o $@ $^ $(CFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
