// -*-c++-*-

/*
 *Copyright:

 Cyrus2D
 Modified by Omid Amini, Nader Zare
 
 Gliders2d
 Modified by Mikhail Prokopenko, Peter Wang

 Copyright (C) Hiroki SHIMORA

 This code is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 3, or (at your option)
 any later version.

 This code is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this code; see the file COPYING.  If not, write to
 the Free Software Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.

 *EndCopyright:
 */

/////////////////////////////////////////////////////////////////////

#ifndef SAMPLE_FIELD_EVALUATOR_H
#define SAMPLE_FIELD_EVALUATOR_H

#include "field_evaluator.h"
#include "predict_state.h"

#include <vector>

namespace rcsc {
class AbstractPlayerObject;
class Vector2D;
}

class ActionStatePair;

class SampleFieldEvaluator
    : public FieldEvaluator {
private:

public:
    SampleFieldEvaluator();

    virtual
    ~SampleFieldEvaluator();

    virtual double operator()(const PredictState &state,
                              const std::vector<ActionStatePair> &path,
                              const rcsc::WorldModel &wm) const;
};

#endif

