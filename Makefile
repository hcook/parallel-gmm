DEPDIR = .dep

TARGET0 = mixtureModelMulticore
CXX = icc
CXXFLAGS = -O2 -ipo -vec-report2
LIBS = -lm
INCLUDES = -I.

DIRS0 = .

SOURCES0= $(foreach dir, $(DIRS0), $(wildcard $(dir)/*.cpp))

OBJS0 = $(SOURCES0:.cpp=.o)

.PHONY: all TAGS clean

all: $(TARGET0)

TAGS:
	@find \( -name '*.c' -o -name '*.cc' -o -name '*.h' \)|etags -l c++ -

clean:
	@rm -rf $(TARGET0) $(OBJS0) $(DEPDIR) `find . \\( -name '*~' \\)`

%.o: %.cpp
	@$(CXX) -MMD -DDEBUG $(CXXFLAGS) $(INCLUDES) -c $< -o $@
	@mkdir -p .dep/$(@D)
	@mv $*.d $(DEPDIR)/$*.P

-include $(SOURCES0:%.cpp=$(DEPDIR)/%.P)

$(TARGET0): $(OBJS0)
	@$(CXX) $(CXXFLAGS) $(OBJS0) $(LIBS) -o $(TARGET0)

