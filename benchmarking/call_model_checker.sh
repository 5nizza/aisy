#!/bin/bash

DIR=`dirname $0`/

# To use blimc bounded model checker run this, where 15 is the bound (change as you wish)
#$DIR/ext_tools/blimc/blimc 15 $1

# To run ic3 run this, note that ic3 we use is (very) slighly changed.
$DIR./ext_tools/ic3-ref/IC3 <$1

