// Dear ImGui: unit tests for ImDrawList batching (vertex offsets, splitter channels, draw callbacks, clip rects).
// Self-contained: builds and runs without a window system, GPU or ImGui context.
//
// Build (from the repository root):
//   clang++ -std=c++11 -I. tests/test_draw_list_batching.cpp imgui.cpp imgui_draw.cpp imgui_tables.cpp imgui_widgets.cpp -o test_draw_list_batching
// or:
//   c++ -std=c++11 -I. tests/test_draw_list_batching.cpp imgui.cpp imgui_draw.cpp imgui_tables.cpp imgui_widgets.cpp -o test_draw_list_batching
// Run:
//   ./test_draw_list_batching
//
// What is being pinned down
// -------------------------
// 1) The offset convention every renderer backend implements (see the "Large meshes" section of
//    imgui.h and backends such as imgui_impl_dx11.cpp / imgui_impl_metal.mm):
//        pcmd->IdxOffset + global_idx_offset, pcmd->VtxOffset + global_vtx_offset
//    i.e. the triangle vertices of a command are
//        VtxBuffer[cmd.VtxOffset + IdxBuffer[cmd.IdxOffset + k]]   for k in [0, cmd.ElemCount)
//    This file reconstructs every triangle through that public convention, so a command that
//    advertises a local index range but whose VtxOffset/IdxOffset points at another vertex
//    segment fails the test even though the index values themselves are in range for a 16-bit
//    ImDrawIdx backend.
// 2) After ImDrawListSplitter::Merge(), commands from later channels are appended (and possibly
//    coalesced) without reordering the content, so the event order observed by a renderer
//    walking CmdBuffer[] is the overlay order: all of channel 0, then all of channel 1, with
//    user callbacks staying exactly where they were submitted.
// 3) Each batch must still carry the clip rectangle that was pushed when its triangles were
//    emitted; a batch whose ClipRect belongs to another channel/scope is a failure.
// 4) A 16-bit index backend (default ImDrawIdx == unsigned short) can express meshes larger than
//    64K vertices as long as the backend honours ImDrawCmd::VtxOffset.

#include "imgui.h"
#include "imgui_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>

//-----------------------------------------------------------------------------
// Minimal assertion harness (no external test framework dependency).
//-----------------------------------------------------------------------------

static int  g_checks_run = 0;
static int  g_checks_failed = 0;
static int  g_checks_printed = 0;

// Large-mesh tests perform millions of checks; if something is systematically wrong, only print the
// first few failures of the current test instead of flooding the log with identical lines.
#define T_MAX_PRINTED_FAILURES 25

static bool ShouldPrintFailure()
{
    if (g_checks_printed >= T_MAX_PRINTED_FAILURES)
        return false;
    g_checks_printed++;
    if (g_checks_printed == T_MAX_PRINTED_FAILURES)
        printf("  ... further failures of this test suppressed.\n");
    return true;
}

#define T_CHECK(EXPR)                                                           \
    do                                                                         \
    {                                                                          \
        g_checks_run++;                                                        \
        if (!(EXPR))                                                           \
        {                                                                      \
            g_checks_failed++;                                                 \
            if (ShouldPrintFailure())                                          \
                printf("  [FAIL] %s:%d: %s\n", __FILE__, __LINE__, #EXPR);     \
        }                                                                      \
    } while (0)

#define T_CHECK_INT(GOT, EXP)                                                   \
    do                                                                         \
    {                                                                          \
        g_checks_run++;                                                        \
        const long long t_got_ = (long long)(GOT);                             \
        const long long t_exp_ = (long long)(EXP);                             \
        if (t_got_ != t_exp_)                                                  \
        {                                                                      \
            g_checks_failed++;                                                 \
            if (ShouldPrintFailure())                                          \
                printf("  [FAIL] %s:%d: %s == %s (got %lld, expected %lld)\n", \
                       __FILE__, __LINE__, #GOT, #EXP, t_got_, t_exp_);        \
        }                                                                      \
    } while (0)

#define T_CHECK_UINT(GOT, EXP)                                                  \
    do                                                                         \
    {                                                                          \
        g_checks_run++;                                                        \
        const unsigned long long t_got_ = (unsigned long long)(GOT);           \
        const unsigned long long t_exp_ = (unsigned long long)(EXP);           \
        if (t_got_ != t_exp_)                                                  \
        {                                                                      \
            g_checks_failed++;                                                 \
            if (ShouldPrintFailure())                                          \
                printf("  [FAIL] %s:%d: %s == %s (got %llu, expected %llu)\n", \
                       __FILE__, __LINE__, #GOT, #EXP, t_got_, t_exp_);        \
        }                                                                      \
    } while (0)

// Exact float comparison is intentional: the tests emit values that are exactly representable
// (small integers plus 0.25) and the draw list copies them verbatim into ImDrawVert.
#define T_CHECK_FLOAT(GOT, EXP)                                                 \
    do                                                                         \
    {                                                                          \
        g_checks_run++;                                                        \
        const double t_got_ = (double)(GOT);                                   \
        const double t_exp_ = (double)(EXP);                                   \
        if (t_got_ != t_exp_)                                                  \
        {                                                                      \
            g_checks_failed++;                                                 \
            if (ShouldPrintFailure())                                          \
                printf("  [FAIL] %s:%d: %s == %s (got %g, expected %g)\n",     \
                       __FILE__, __LINE__, #GOT, #EXP, t_got_, t_exp_);        \
        }                                                                      \
    } while (0)

//-----------------------------------------------------------------------------
// Fixture: a standalone ImDrawList fed by a standalone ImDrawListSharedData.
// No ImGui context, no window, no font atlas: this is a pure unit test of ImDrawList.
//-----------------------------------------------------------------------------

struct TestDrawList
{
    ImDrawListSharedData    SharedData;
    ImDrawList              DrawList;

    explicit TestDrawList(ImDrawFlags initial_flags = ImDrawFlags_UseVtxOffset)
        : DrawList(&SharedData)
    {
        SharedData.InitialDrawFlags = initial_flags;
        SharedData.ClipRectFullscreen = ImVec4(0.0f, 0.0f, 4096.0f, 4096.0f);
        DrawList._ResetForNewFrame();
        DrawList._SetPixelDensity(1.0f);
        DrawList.PushClipRectFullScreen();
    }

    // Every quad emitted here is fully described by its id:
    //   position:  a = (id, 0), b = (id + 0.25, 0), c = (id + 0.25, 1), d = (id, 1)
    //   uv      : (id, 0), (id, 1), (id, 2), (id, 3)
    //   color   : 0xFF000000 | id
    // PrimQuadUV() emits indices {0,1,2} and {0,2,3} relative to its four vertices, i.e. this
    // also pins down the index order inside a quad.
    void EmitQuad(ImU32 id)
    {
        const ImVec2 a((float)id, 0.0f);
        const ImVec2 b((float)id + 0.25f, 0.0f);
        const ImVec2 c((float)id + 0.25f, 1.0f);
        const ImVec2 d((float)id, 1.0f);
        DrawList.PrimReserve(6, 4);
        DrawList.PrimQuadUV(a, b, c, d,
                            ImVec2((float)id, 0.0f), ImVec2((float)id, 1.0f),
                            ImVec2((float)id, 2.0f), ImVec2((float)id, 3.0f),
                            0xFF000000u | id);
    }
};

struct QuadSpec
{
    unsigned    Id;
    ImVec4      Clip;
};

struct Event
{
    enum KindEnum { Quad, Callback };
    KindEnum    Kind;
    unsigned    Id;                 // quad id, or callback payload id

    static Event QuadEvent(unsigned id)      { Event e; e.Kind = Quad; e.Id = id; return e; }
    static Event CallbackEvent(unsigned id)  { Event e; e.Kind = Callback; e.Id = id; return e; }
};

static const char* EventKindName(Event::KindEnum kind)
{
    return kind == Event::Quad ? "quad" : "callback";
}

//-----------------------------------------------------------------------------
// Backend-side resolution helpers: this is the only way these tests are allowed to look at
// geometry, mirroring what a renderer backend does with ImDrawCmd::IdxOffset/VtxOffset.
//-----------------------------------------------------------------------------

// A real ImDrawCallback, invoked by the walk below exactly like a backend invokes it. Each test
// passes a distinct payload so the invocation order is observable from inside the callback.
static std::vector<Event>* g_walk_events = NULL;

static void RecordingCallback(const ImDrawList* parent_list, const ImDrawCmd* cmd)
{
    IM_UNUSED(parent_list);
    const unsigned payload = (cmd != NULL && cmd->UserCallbackData != NULL) ? *(const unsigned*)cmd->UserCallbackData : 0xFFFFFFFFu;
    if (g_walk_events != NULL)
        g_walk_events->push_back(Event::CallbackEvent(payload));
}

static unsigned BackendLocalIndex(const ImDrawList& draw_list, const ImDrawCmd& cmd, unsigned slot)
{
    return (unsigned)draw_list.IdxBuffer[(int)cmd.IdxOffset + (int)slot];
}

static const ImDrawVert& BackendVertex(const ImDrawList& draw_list, const ImDrawCmd& cmd, unsigned local_index)
{
    return draw_list.VtxBuffer[(int)cmd.VtxOffset + (int)local_index];
}

// Verify one reconstructed vertex against the quad corner it is supposed to represent.
static void CheckReconstructedVertex(const ImDrawVert& v, unsigned quad_id, int corner)
{
    static const float kCornerX[4] = { 0.0f, 0.25f, 0.25f, 0.0f };
    static const float kCornerY[4] = { 0.0f, 0.0f, 1.0f, 1.0f };

    T_CHECK_UINT(v.col, 0xFF000000u | quad_id);
    T_CHECK_FLOAT(v.uv.x, (float)quad_id);
    T_CHECK_FLOAT(v.uv.y, (float)corner);
    T_CHECK_FLOAT(v.pos.x, (float)quad_id + kCornerX[corner]);
    T_CHECK_FLOAT(v.pos.y, kCornerY[corner]);
}

// Walk the draw list the way a renderer does and validate every triangle:
//  - user callbacks are invoked in place (copying the backend contract: call cmd.UserCallback
//    instead of rendering triangles), so the recorded event order is what the application
//    callback really observes relative to the surrounding draws;
//  - indices are resolved through cmd.IdxOffset + cmd.VtxOffset only;
//  - each 6-index batch is one quad and all six vertices must describe that same quad;
//  - every index fits in the compiled ImDrawIdx type (when 16-bit) and resolves to a vertex;
//  - the command ClipRect is the clip that was active when the quads were emitted.
static void WalkAndValidate(const ImDrawList& draw_list,
                            const std::vector<QuadSpec>& quads,
                            std::vector<Event>* out_events)
{
    g_walk_events = out_events;
    for (int cmd_n = 0; cmd_n < draw_list.CmdBuffer.Size; cmd_n++)
    {
        const ImDrawCmd& cmd = draw_list.CmdBuffer[cmd_n];

        if (cmd.UserCallback != NULL)
        {
            // A callback is a barrier, never a draw: it must not consume indices/vertices.
            T_CHECK_UINT(cmd.ElemCount, 0u);
            T_CHECK(cmd.VtxOffset <= (unsigned)draw_list.VtxBuffer.Size);
            T_CHECK(cmd.IdxOffset <= (unsigned)draw_list.IdxBuffer.Size);
            cmd.UserCallback(&draw_list, &cmd); // Records one Callback event.
            continue;
        }

        if (cmd.ElemCount == 0)
            continue; // Trailing command slot kept by the draw list, not visible to the backend.

        // Framebuffer-space sanity of the batch clip region.
        T_CHECK(cmd.ClipRect.x <= cmd.ClipRect.z);
        T_CHECK(cmd.ClipRect.y <= cmd.ClipRect.w);

        // Index range advertised by this command must be inside the buffers.
        T_CHECK(cmd.IdxOffset + cmd.ElemCount <= (unsigned)draw_list.IdxBuffer.Size);
        T_CHECK(cmd.VtxOffset <= (unsigned)draw_list.VtxBuffer.Size);

        // These tests only emit whole quads: 6 indices (2 triangles) per quad.
        T_CHECK_UINT(cmd.ElemCount % 6u, 0u);

        for (unsigned e = 0; e + 5 < cmd.ElemCount; e += 6)
        {
            unsigned local[6];
            for (unsigned k = 0; k < 6; k++)
            {
                local[k] = BackendLocalIndex(draw_list, cmd, e + k);

                // A 16-bit ImDrawIdx must be able to address this local index.
                if (sizeof(ImDrawIdx) == 2)
                    T_CHECK(local[k] < (1u << 16));

                // Vertex must exist once VtxOffset resolves the local index.
                T_CHECK(cmd.VtxOffset + local[k] < (unsigned)draw_list.VtxBuffer.Size);
            }

            // Index shape of PrimQuadUV(): {base+0, base+1, base+2} and {base+0, base+2, base+3}.
            T_CHECK_UINT(local[1], local[0] + 1u);
            T_CHECK_UINT(local[2], local[0] + 2u);
            T_CHECK_UINT(local[3], local[0]);
            T_CHECK_UINT(local[4], local[0] + 2u);
            T_CHECK_UINT(local[5], local[0] + 3u);

            const ImDrawVert& v0 = BackendVertex(draw_list, cmd, local[0]);
            const ImDrawVert& v1 = BackendVertex(draw_list, cmd, local[1]);
            const ImDrawVert& v2 = BackendVertex(draw_list, cmd, local[2]);
            const ImDrawVert& v3 = BackendVertex(draw_list, cmd, local[3]);
            const ImDrawVert& v4 = BackendVertex(draw_list, cmd, local[4]);
            const ImDrawVert& v5 = BackendVertex(draw_list, cmd, local[5]);

            // All six vertices must describe the same quad, identifiable directly from the data
            // (a wrong VtxOffset resolves the local indices to another vertex segment and fails here).
            const unsigned quad_id = v0.col & 0xFFFFFFu;
            T_CHECK(quad_id < quads.size());
            T_CHECK(v1.col == v0.col && v2.col == v0.col && v3.col == v0.col && v4.col == v0.col && v5.col == v0.col);
            if (quad_id >= quads.size())
                continue; // Cannot describe this quad; already reported above.

            CheckReconstructedVertex(v0, quad_id, 0);
            CheckReconstructedVertex(v1, quad_id, 1);
            CheckReconstructedVertex(v2, quad_id, 2);
            CheckReconstructedVertex(v3, quad_id, 0);
            CheckReconstructedVertex(v4, quad_id, 2);
            CheckReconstructedVertex(v5, quad_id, 3);

            // The batch clip must be the clip that was pushed for that quad.
            T_CHECK(cmd.ClipRect.x == quads[quad_id].Clip.x);
            T_CHECK(cmd.ClipRect.y == quads[quad_id].Clip.y);
            T_CHECK(cmd.ClipRect.z == quads[quad_id].Clip.z);
            T_CHECK(cmd.ClipRect.w == quads[quad_id].Clip.w);

            if (out_events != NULL)
                out_events->push_back(Event::QuadEvent(quad_id));
        }
    }
    g_walk_events = NULL;
}

static void CheckEventSequence(const char* label,
                               const std::vector<Event>& expected,
                               const std::vector<Event>& actual)
{
    T_CHECK_INT(actual.size(), expected.size());
    const size_t count = expected.size() < actual.size() ? expected.size() : actual.size();
    for (size_t n = 0; n < count; n++)
    {
        T_CHECK(expected[n].Kind == actual[n].Kind);
        T_CHECK_UINT(actual[n].Id, expected[n].Id);
        if (expected[n].Kind != actual[n].Kind || expected[n].Id != actual[n].Id)
        {
            printf("  [FAIL] %s: event #%d is %s(%u), expected %s(%u)\n", label, (int)n,
                   EventKindName(actual[n].Kind), actual[n].Id,
                   EventKindName(expected[n].Kind), expected[n].Id);
            return;
        }
    }
}

// All quads exactly once, in the expected order, and no gaps in IdxOffset between draw commands.
static void CheckCoverage(const ImDrawList& draw_list, unsigned quad_count,
                          const std::vector<Event>& events)
{
    std::vector<unsigned> seen(quad_count, 0);
    for (size_t n = 0; n < events.size(); n++)
    {
        if (events[n].Kind != Event::Quad)
            continue;
        T_CHECK(events[n].Id < quad_count);
        if (events[n].Id < quad_count)
        {
            seen[events[n].Id]++;
            T_CHECK_UINT(seen[events[n].Id], 1u);
        }
    }
    unsigned missing = 0;
    for (unsigned n = 0; n < quad_count; n++)
        if (seen[n] == 0)
            missing++;
    T_CHECK_UINT(missing, 0u);

    // Draw commands must tile IdxBuffer without holes: the backend reads exactly
    // ElemCount indices at IdxOffset.
    unsigned idx_cursor = 0;
    for (int cmd_n = 0; cmd_n < draw_list.CmdBuffer.Size; cmd_n++)
    {
        const ImDrawCmd& cmd = draw_list.CmdBuffer[cmd_n];
        if (cmd.UserCallback != NULL || cmd.ElemCount == 0)
            continue;
        T_CHECK_UINT(cmd.IdxOffset, idx_cursor);
        idx_cursor += cmd.ElemCount;
    }
    T_CHECK_UINT(idx_cursor, (unsigned)draw_list.IdxBuffer.Size);
}

// Walk a draw list after it was added to ImDrawData (which resolves callback data pointers the
// same way a backend sees them).
static void WalkDrawDataAndValidate(const char* label, const ImDrawData& draw_data,
                                    const std::vector<QuadSpec>& quads, std::vector<Event>* out_events)
{
    IM_UNUSED(label);
    T_CHECK(draw_data.CmdLists.Size >= 1);
    if (draw_data.CmdLists.Size < 1)
        return;
    WalkAndValidate(*draw_data.CmdLists[0], quads, out_events);
}

//-----------------------------------------------------------------------------
// Test 1: 16-bit indices + VtxOffset express a >64K vertex mesh.
//
// 20000 quads == 80000 vertices. PrimReserve() must start a new vertex segment (VtxOffset > 0)
// once the local index would exceed 16 bits, resetting local indices to 0. Both segments must
// therefore be readable through VtxOffset, and local index 0 of the second segment must map to
// a different vertex than local index 0 of the first segment.
//-----------------------------------------------------------------------------

static void Test_SmallIndex_LargeMesh_VtxOffset()
{
    const unsigned kQuadCount = 20000;
    const ImVec4 kClip(10.0f, 20.0f, 110.0f, 120.0f);

    TestDrawList fixture;
    ImDrawList& dl = fixture.DrawList;
    T_CHECK((dl.Flags & ImDrawFlags_UseVtxOffset) != 0);
    dl.PushClipRect(ImVec2(kClip.x, kClip.y), ImVec2(kClip.z, kClip.w), false);
    for (unsigned id = 0; id < kQuadCount; id++)
        fixture.EmitQuad(id);
    dl.PopClipRect();

    std::vector<QuadSpec> quads(kQuadCount);
    for (unsigned id = 0; id < kQuadCount; id++)
    {
        quads[id].Id = id;
        quads[id].Clip = kClip;
    }

    // The whole mesh (all 4 vertices of every quad) is stored.
    T_CHECK_INT(dl.VtxBuffer.Size, (int)(kQuadCount * 4));
    T_CHECK_INT(dl.IdxBuffer.Size, (int)(kQuadCount * 6));

    // Find how many commands hold geometry and whether VtxOffset actually kicked in.
    int draw_cmds = 0;
    int offset_cmds = 0;
    int first_offset_cmd = -1;
    for (int cmd_n = 0; cmd_n < dl.CmdBuffer.Size; cmd_n++)
    {
        const ImDrawCmd& cmd = dl.CmdBuffer[cmd_n];
        if (cmd.UserCallback != NULL || cmd.ElemCount == 0)
            continue;
        draw_cmds++;
        if (cmd.VtxOffset > 0)
        {
            offset_cmds++;
            if (first_offset_cmd < 0)
                first_offset_cmd = cmd_n;
        }
    }

    if (sizeof(ImDrawIdx) == 2)
    {
        // A 16-bit index backend needs at least a second segment to hold 80K vertices...
        T_CHECK(draw_cmds >= 2);
        T_CHECK(offset_cmds >= 1);
        T_CHECK(first_offset_cmd >= 0);
    }

    if (first_offset_cmd >= 0)
    {
        const ImDrawCmd& first_offset = dl.CmdBuffer[first_offset_cmd];
        const unsigned base = first_offset.VtxOffset;

        // VtxOffset points at a whole quad boundary of the vertex buffer.
        T_CHECK_UINT(base % 4u, 0u);
        T_CHECK(base < (unsigned)dl.VtxBuffer.Size);

        // Local indices restart from 0 in the new segment...
        T_CHECK_UINT(BackendLocalIndex(dl, first_offset, 0), 0u);

        // ...and the same local index 0 in the previous segment designates another vertex.
        // Without honouring VtxOffset, the first triangle of the new segment would reconstruct
        // quad 0 (magenta) instead of the right quad, which is exactly the failure this test
        // is designed to detect.
        T_CHECK(dl.VtxBuffer[0].col != dl.VtxBuffer[base].col);
        T_CHECK_UINT(dl.VtxBuffer[base].col & 0xFFFFFFu, base / 4u);
        T_CHECK_UINT(BackendVertex(dl, first_offset, 0).col & 0xFFFFFFu, base / 4u);

        // The preceding command must not have used VtxOffset (its indices are self-contained).
        int previous_draw = -1;
        for (int cmd_n = first_offset_cmd - 1; cmd_n >= 0; cmd_n--)
            if (dl.CmdBuffer[cmd_n].UserCallback == NULL && dl.CmdBuffer[cmd_n].ElemCount != 0)
            {
                previous_draw = cmd_n;
                break;
            }
        if (previous_draw >= 0)
            T_CHECK_UINT(dl.CmdBuffer[previous_draw].VtxOffset, 0u);
    }

    std::vector<Event> expected;
    for (unsigned id = 0; id < kQuadCount; id++)
        expected.push_back(Event::QuadEvent(id));

    std::vector<Event> actual;
    WalkAndValidate(dl, quads, &actual);
    CheckEventSequence("SmallIndex_LargeMesh", expected, actual);
    CheckCoverage(dl, kQuadCount, actual);
}

//-----------------------------------------------------------------------------
// Test 2: splitter channel merge keeps the overlay order and the callback barrier.
//
// Two channels. Channel 0 has clip A at its head, a user callback in the middle, and its tail
// under the same clip A. Channel 1 starts again under clip A and ends under clip B. After
// Merge() the renderer must see:
//     quad 0, callback, quad 1, quad 2, quad 3
// with clips A,A,A,A,B. The callback must be a zero-element command between the command that
// draws quad 0 and the command that draws quad 1, and its clip is A. Handing the merged list
// through ImDrawData::AddDrawList() must resolve the stored callback payload as documented.
//-----------------------------------------------------------------------------

static void Test_Splitter_CallbackOrderAndClips()
{
    const ImVec4 kClipA(0.0f, 0.0f, 100.0f, 100.0f);
    const ImVec4 kClipB(200.0f, 200.0f, 300.0f, 300.0f);

    TestDrawList fixture;
    ImDrawList& dl = fixture.DrawList;
    ImDrawListSplitter splitter;
    splitter.Split(&dl, 2);

    static unsigned s_callback_payload = 7;

    // Channel 0: head draw, callback, tail draw -- all under clip A.
    splitter.SetCurrentChannel(&dl, 0);
    dl.PushClipRect(ImVec2(kClipA.x, kClipA.y), ImVec2(kClipA.z, kClipA.w), false);
    fixture.EmitQuad(0);
    dl.AddCallback(RecordingCallback, &s_callback_payload, sizeof(s_callback_payload));
    fixture.EmitQuad(1);
    dl.PopClipRect();

    // Channel 1: same clip A at its head, clip B for its tail.
    splitter.SetCurrentChannel(&dl, 1);
    dl.PushClipRect(ImVec2(kClipA.x, kClipA.y), ImVec2(kClipA.z, kClipA.w), false);
    fixture.EmitQuad(2);
    dl.PopClipRect();
    dl.PushClipRect(ImVec2(kClipB.x, kClipB.y), ImVec2(kClipB.z, kClipB.w), false);
    fixture.EmitQuad(3);
    dl.PopClipRect();

    splitter.Merge(&dl);

    std::vector<QuadSpec> quads(4);
    quads[0].Id = 0; quads[0].Clip = kClipA;
    quads[1].Id = 1; quads[1].Clip = kClipA;
    quads[2].Id = 2; quads[2].Clip = kClipA;
    quads[3].Id = 3; quads[3].Clip = kClipB;

    // Callback must be preserved, alone in its command and with clip A.
    int callback_cmd = -1;
    int callback_count = 0;
    for (int cmd_n = 0; cmd_n < dl.CmdBuffer.Size; cmd_n++)
        if (dl.CmdBuffer[cmd_n].UserCallback != NULL)
        {
            callback_count++;
            callback_cmd = cmd_n;
            T_CHECK_UINT(dl.CmdBuffer[cmd_n].ElemCount, 0u);
            T_CHECK_FLOAT(dl.CmdBuffer[cmd_n].ClipRect.x, kClipA.x);
            T_CHECK_FLOAT(dl.CmdBuffer[cmd_n].ClipRect.y, kClipA.y);
            T_CHECK_FLOAT(dl.CmdBuffer[cmd_n].ClipRect.z, kClipA.z);
            T_CHECK_FLOAT(dl.CmdBuffer[cmd_n].ClipRect.w, kClipA.w);
        }
    T_CHECK_INT(callback_count, 1);
    T_CHECK(callback_cmd >= 0);

    // Locate the draw commands around the callback by resolving their first quad through the
    // backend convention.
    int quad0_cmd = -1, quad1_cmd = -1;
    for (int cmd_n = 0; cmd_n < dl.CmdBuffer.Size; cmd_n++)
    {
        const ImDrawCmd& cmd = dl.CmdBuffer[cmd_n];
        if (cmd.UserCallback != NULL || cmd.ElemCount == 0)
            continue;
        const unsigned id = BackendVertex(dl, cmd, BackendLocalIndex(dl, cmd, 0)).col & 0xFFFFFFu;
        if (id == 0)
            quad0_cmd = cmd_n;
        if (id == 1 && quad1_cmd < 0)
            quad1_cmd = cmd_n;
    }
    T_CHECK(quad0_cmd >= 0);
    T_CHECK(quad1_cmd >= 0);
    T_CHECK(quad0_cmd < callback_cmd);
    T_CHECK(callback_cmd < quad1_cmd);

    // Hand the merged list to the documented draw-data path: this resolves the stored callback
    // payload into the draw list's internal buffer.
    ImDrawData draw_data;
    draw_data.AddDrawList(&dl);

    T_CHECK_INT(draw_data.CmdLists.Size, 1);
    T_CHECK_INT(draw_data.TotalVtxCount, 16);
    T_CHECK_INT(draw_data.TotalIdxCount, 24);

    const ImDrawCmd& resolved_cb = dl.CmdBuffer[callback_cmd];
    T_CHECK(resolved_cb.UserCallbackData != NULL);
    if (resolved_cb.UserCallbackData != NULL)
    {
        // userdata_size > 0 means the payload is copied into the list, so the visible pointer must
        // not alias the caller's variable but carry the same bytes.
        T_CHECK(resolved_cb.UserCallbackData != (const void*)&s_callback_payload);
        T_CHECK_UINT(*(const unsigned*)resolved_cb.UserCallbackData, s_callback_payload);
    }
    T_CHECK_INT(resolved_cb.UserCallbackDataSize, (int)sizeof(s_callback_payload));

    std::vector<Event> expected;
    expected.push_back(Event::QuadEvent(0));
    expected.push_back(Event::CallbackEvent(s_callback_payload));
    expected.push_back(Event::QuadEvent(1));
    expected.push_back(Event::QuadEvent(2));
    expected.push_back(Event::QuadEvent(3));

    std::vector<Event> actual;
    WalkDrawDataAndValidate("Splitter_CallbackOrder", draw_data, quads, &actual);
    CheckEventSequence("Splitter_CallbackOrder", expected, actual);
    CheckCoverage(dl, 4, actual);
}

//-----------------------------------------------------------------------------
// Test 3: splitter merge with a >64K vertex mesh spread over two channels.
//
// Both channels contain enough quads that the draw list must restart a 16-bit vertex segment in
// the middle of a channel. After merging, all triangles must still resolve through
// (IdxOffset, VtxOffset), the callback submitted mid-channel must stay between its neighbours,
// and clip regions per channel must remain attached to the right batches.
//-----------------------------------------------------------------------------

static void Test_Splitter_LargeMesh_CallbackOrder()
{
    // 12000 quads per channel == 48000 vertices per channel, 96000 vertices total.
    const unsigned kQuadsPerChannel = 12000;
    const unsigned kCallbackAt = 10000;
    const ImVec4 kClipA(0.0f, 0.0f, 640.0f, 480.0f);
    const ImVec4 kClipB(32.0f, 32.0f, 800.0f, 600.0f);

    TestDrawList fixture;
    ImDrawList& dl = fixture.DrawList;
    ImDrawListSplitter splitter;
    splitter.Split(&dl, 2);

    static unsigned s_callback_payload = 3;

    splitter.SetCurrentChannel(&dl, 0);
    dl.PushClipRect(ImVec2(kClipA.x, kClipA.y), ImVec2(kClipA.z, kClipA.w), false);
    for (unsigned id = 0; id < kQuadsPerChannel; id++)
    {
        fixture.EmitQuad(id);
        if (id + 1 == kCallbackAt)
            dl.AddCallback(RecordingCallback, &s_callback_payload, 0);
    }
    dl.PopClipRect();

    splitter.SetCurrentChannel(&dl, 1);
    dl.PushClipRect(ImVec2(kClipB.x, kClipB.y), ImVec2(kClipB.z, kClipB.w), false);
    for (unsigned id = kQuadsPerChannel; id < kQuadsPerChannel * 2; id++)
        fixture.EmitQuad(id);
    dl.PopClipRect();

    splitter.Merge(&dl);

    const unsigned kQuadCount = kQuadsPerChannel * 2;
    std::vector<QuadSpec> quads(kQuadCount);
    for (unsigned id = 0; id < kQuadCount; id++)
    {
        quads[id].Id = id;
        quads[id].Clip = (id < kQuadsPerChannel) ? kClipA : kClipB;
    }

    T_CHECK_INT(dl.VtxBuffer.Size, (int)(kQuadCount * 4));

    int offset_cmds = 0;
    for (int cmd_n = 0; cmd_n < dl.CmdBuffer.Size; cmd_n++)
    {
        const ImDrawCmd& cmd = dl.CmdBuffer[cmd_n];
        if (cmd.UserCallback == NULL && cmd.ElemCount != 0 && cmd.VtxOffset > 0)
            offset_cmds++;
    }
    if (sizeof(ImDrawIdx) == 2)
        T_CHECK(offset_cmds >= 1);

    // The callback must be between quad kCallbackAt-1 and quad kCallbackAt.
    int last_before_callback = -1, callback_cmd = -1, first_after_callback = -1;
    for (int cmd_n = 0; cmd_n < dl.CmdBuffer.Size; cmd_n++)
    {
        const ImDrawCmd& cmd = dl.CmdBuffer[cmd_n];
        if (cmd.UserCallback != NULL)
        {
            T_CHECK_INT(callback_cmd, -1);
            callback_cmd = cmd_n;
            T_CHECK_UINT(cmd.ElemCount, 0u);
            continue;
        }
        if (cmd.ElemCount == 0)
            continue;
        const unsigned first_id = BackendVertex(dl, cmd, BackendLocalIndex(dl, cmd, 0)).col & 0xFFFFFFu;
        const unsigned last_local = BackendLocalIndex(dl, cmd, cmd.ElemCount - 6);
        const unsigned last_id = BackendVertex(dl, cmd, last_local).col & 0xFFFFFFu;
        if (callback_cmd < 0)
            last_before_callback = (int)last_id;
        else if (first_after_callback < 0)
            first_after_callback = (int)first_id;
    }
    T_CHECK(callback_cmd >= 0);
    T_CHECK_INT(last_before_callback, (int)(kCallbackAt - 1));
    T_CHECK_INT(first_after_callback, (int)kCallbackAt);

    std::vector<Event> expected;
    for (unsigned id = 0; id < kCallbackAt; id++)
        expected.push_back(Event::QuadEvent(id));
    expected.push_back(Event::CallbackEvent(s_callback_payload));
    for (unsigned id = kCallbackAt; id < kQuadCount; id++)
        expected.push_back(Event::QuadEvent(id));

    std::vector<Event> actual;
    WalkAndValidate(dl, quads, &actual);
    CheckEventSequence("Splitter_LargeMesh", expected, actual);
    CheckCoverage(dl, kQuadCount, actual);
}

//-----------------------------------------------------------------------------
// Test 4: clip-region changes split batches, and merging channels never reorders geometry.
//
// Three quads in one channel with clips A, B, A. The draw list must emit batches whose ClipRect
// matches the clip in effect when their quads were added (the two A batches are not allowed to
// swallow the B quad), and the reconstructed order must be 0,1,2. A second channel then adds
// quad 3 under A; after merge that quad must come last even though its clip matches earlier
// geometry, i.e. batches may merge only without changing the overlay order.
//-----------------------------------------------------------------------------

static void Test_ClipChangeAndMergeDoesNotReorder()
{
    const ImVec4 kClipA(0.0f, 0.0f, 100.0f, 100.0f);
    const ImVec4 kClipB(50.0f, 50.0f, 150.0f, 150.0f);

    TestDrawList fixture;
    ImDrawList& dl = fixture.DrawList;
    ImDrawListSplitter splitter;
    splitter.Split(&dl, 2);

    splitter.SetCurrentChannel(&dl, 0);
    dl.PushClipRect(ImVec2(kClipA.x, kClipA.y), ImVec2(kClipA.z, kClipA.w), false);
    fixture.EmitQuad(0);
    dl.PopClipRect();
    dl.PushClipRect(ImVec2(kClipB.x, kClipB.y), ImVec2(kClipB.z, kClipB.w), false);
    fixture.EmitQuad(1);
    dl.PopClipRect();
    dl.PushClipRect(ImVec2(kClipA.x, kClipA.y), ImVec2(kClipA.z, kClipA.w), false);
    fixture.EmitQuad(2);
    dl.PopClipRect();

    splitter.SetCurrentChannel(&dl, 1);
    dl.PushClipRect(ImVec2(kClipA.x, kClipA.y), ImVec2(kClipA.z, kClipA.w), false);
    fixture.EmitQuad(3);
    dl.PopClipRect();

    splitter.Merge(&dl);

    std::vector<QuadSpec> quads(4);
    quads[0].Id = 0; quads[0].Clip = kClipA;
    quads[1].Id = 1; quads[1].Clip = kClipB;
    quads[2].Id = 2; quads[2].Clip = kClipA;
    quads[3].Id = 3; quads[3].Clip = kClipA;

    std::vector<Event> expected;
    expected.push_back(Event::QuadEvent(0));
    expected.push_back(Event::QuadEvent(1));
    expected.push_back(Event::QuadEvent(2));
    expected.push_back(Event::QuadEvent(3));

    std::vector<Event> actual;
    WalkAndValidate(dl, quads, &actual);
    CheckEventSequence("ClipChangeAndMerge", expected, actual);
    CheckCoverage(dl, 4, actual);

    // The two A quads around the B quad cannot share a command; the B quad needs its own batch.
    // (Exact command count is implementation-defined, but every command must hold exactly one
    // clip and non-empty batches must exist.)
    int non_empty = 0;
    for (int cmd_n = 0; cmd_n < dl.CmdBuffer.Size; cmd_n++)
        if (dl.CmdBuffer[cmd_n].UserCallback == NULL && dl.CmdBuffer[cmd_n].ElemCount != 0)
            non_empty++;
    T_CHECK(non_empty >= 3);
}

//-----------------------------------------------------------------------------
// Test 5: same geometry with VtxOffset disabled (baseline for small meshes).
//
// Ensures the validation helpers themselves are not silently relying on VtxOffset being present:
// a small list with ImDrawFlags_None must pass the same reconstructed-triangle checks with all
// VtxOffset values at 0, and still honour clip changes and callbacks across channel merging.
//-----------------------------------------------------------------------------

static void Test_NoVtxOffset_Baseline()
{
    const ImVec4 kClip(16.0f, 16.0f, 816.0f, 616.0f);

    TestDrawList fixture(ImDrawFlags_None);
    ImDrawList& dl = fixture.DrawList;
    T_CHECK_INT((int)(dl.Flags & ImDrawFlags_UseVtxOffset), 0);

    static unsigned s_callback_payload = 11;
    ImDrawListSplitter splitter;
    splitter.Split(&dl, 2);

    splitter.SetCurrentChannel(&dl, 0);
    dl.PushClipRect(ImVec2(kClip.x, kClip.y), ImVec2(kClip.z, kClip.w), false);
    fixture.EmitQuad(0);
    dl.AddCallback(RecordingCallback, &s_callback_payload, 0);
    fixture.EmitQuad(1);

    splitter.SetCurrentChannel(&dl, 1);
    fixture.EmitQuad(2);
    dl.PopClipRect();
    splitter.Merge(&dl);

    for (int cmd_n = 0; cmd_n < dl.CmdBuffer.Size; cmd_n++)
        if (dl.CmdBuffer[cmd_n].ElemCount != 0)
            T_CHECK_UINT(dl.CmdBuffer[cmd_n].VtxOffset, 0u);

    std::vector<QuadSpec> quads(3);
    for (unsigned id = 0; id < 3; id++)
    {
        quads[id].Id = id;
        quads[id].Clip = kClip;
    }

    std::vector<Event> expected;
    expected.push_back(Event::QuadEvent(0));
    expected.push_back(Event::CallbackEvent(s_callback_payload));
    expected.push_back(Event::QuadEvent(1));
    expected.push_back(Event::QuadEvent(2));

    std::vector<Event> actual;
    WalkAndValidate(dl, quads, &actual);
    CheckEventSequence("NoVtxOffset_Baseline", expected, actual);
    CheckCoverage(dl, 3, actual);
}

//-----------------------------------------------------------------------------
// Test 6: a channel switch must respect VtxOffset when deciding whether adjacent commands
// can be coalesced.
//
// Channel 0 is drawn, then channel 1 starts while the vertex segment is still the first one
// (VtxOffset == 0). Channel 0 then continues long enough to start a second 16-bit vertex segment
// (VtxOffset > 0). After Merge(), channel 1's first command does not use VtxOffset while the
// command before it does. If the merge/command-merging logic compares only ClipRect+TexRef (not
// VtxOffset), the two get coalesced and channel 1's first batch inherits channel 0's VtxOffset:
// its local index 0 then resolves to a completely different quad. The test pins down both the
// batch split and the resolved geometry.
//-----------------------------------------------------------------------------

static void Test_Splitter_ChannelSwitchRespectsVtxOffset()
{
    // Channel 0 grows past the 16-bit boundary while drawing; channel 1 holds a single quad that
    // was emitted *before* the crossing. That quad lives in the first vertex segment
    // (VtxOffset == 0) whereas channel 0's last command lives in the second segment
    // (VtxOffset > 0). Their ClipRect and texture match, so only VtxOffset tells them apart.
    const unsigned kChannel0Quads = 16400; // 16400 quads == 65600 vertices > 64K
    const ImVec4 kClip(0.0f, 0.0f, 128.0f, 128.0f);

    TestDrawList fixture;
    ImDrawList& dl = fixture.DrawList;
    ImDrawListSplitter splitter;
    splitter.Split(&dl, 2);

    // One clip for the whole test so every quad shares the same ClipRect/TexRef: the point here is
    // the VtxOffset bookkeeping across channel switches, not clip changes.
    dl.PushClipRect(ImVec2(kClip.x, kClip.y), ImVec2(kClip.z, kClip.w), false);

    // Ids are dense across the test: 0 is the first channel-0 quad, 1 is the channel-1 quad,
    // 2..kChannel0Quads-1 are the remaining channel-0 quads.
    splitter.SetCurrentChannel(&dl, 0);
    fixture.EmitQuad(0); // VtxOffset == 0

    splitter.SetCurrentChannel(&dl, 1);
    fixture.EmitQuad(1); // channel 1 content emitted while VtxOffset is still 0

    splitter.SetCurrentChannel(&dl, 0);
    for (unsigned id = 2; id < kChannel0Quads; id++)
        fixture.EmitQuad(id); // channel 0 crosses 64K vertices -> second segment

    dl.PopClipRect();
    splitter.Merge(&dl);

    const unsigned kQuadCount = kChannel0Quads;
    std::vector<QuadSpec> quads(kQuadCount);
    for (unsigned id = 0; id < kQuadCount; id++)
    {
        quads[id].Id = id;
        quads[id].Clip = kClip;
    }

    if (sizeof(ImDrawIdx) == 2)
    {
        // Locate channel 1's quad (id 1) and the command immediately before it after merging.
        int channel1_cmd = -1;
        for (int cmd_n = 0; cmd_n < dl.CmdBuffer.Size; cmd_n++)
        {
            const ImDrawCmd& cmd = dl.CmdBuffer[cmd_n];
            if (cmd.UserCallback != NULL || cmd.ElemCount == 0)
                continue;
            const unsigned id = BackendVertex(dl, cmd, BackendLocalIndex(dl, cmd, 0)).col & 0xFFFFFFu;
            if (id == 1)
                channel1_cmd = cmd_n;
        }
        T_CHECK(channel1_cmd >= 0);
        if (channel1_cmd > 0)
        {
            const ImDrawCmd& previous = dl.CmdBuffer[channel1_cmd - 1];
            if (previous.UserCallback == NULL && previous.ElemCount != 0)
            {
                // Channel 0's tail straddles into the second vertex segment. Since ClipRect/TexRef
                // match, a merge that ignores VtxOffset would fuse these two commands and make
                // quad 1 resolve into channel 0's second segment. They must stay distinct.
                T_CHECK_UINT(dl.CmdBuffer[channel1_cmd].VtxOffset, 0u);
                T_CHECK(previous.VtxOffset > 0u);
                T_CHECK(previous.VtxOffset != dl.CmdBuffer[channel1_cmd].VtxOffset);
            }
        }
    }

    // Full reconstruction: every quad resolves through its own command's VtxOffset. A bad coalesce
    // makes quad 1 resolve to another quad and this walk catches it.
    std::vector<Event> actual;
    WalkAndValidate(dl, quads, &actual);

    // Merge order is channel 0 (submission order: 0, 2, 3, ..., kChannel0Quads-1) then channel 1 (1).
    std::vector<Event> expected;
    expected.push_back(Event::QuadEvent(0));
    for (unsigned id = 2; id < kChannel0Quads; id++)
        expected.push_back(Event::QuadEvent(id));
    expected.push_back(Event::QuadEvent(1));

    CheckEventSequence("ChannelSwitchRespectsVtxOffset", expected, actual);
    CheckCoverage(dl, kQuadCount, actual);
}

//-----------------------------------------------------------------------------
// Test 7: multi-channel overlay ordering.
//
// Real overlays (tables/columns) split into many channels and interleave callbacks, so the merge
// order across more than two channels matters. The quads are submitted in an interleaved order:
//     ch0: 0, 3, 6   ch1: 1, 4, 7   ch2: 2, 5, 8
// A correct Merge() must produce 0,1,2,3,4,5,6,7,8 (all of channel 0, then channel 1, then
// channel 2), not the submission order and not the reverse channel order. Two callbacks pin the
// channel boundaries: one inside channel 1 (between quads 4 and 7) and one inside channel 2
// (between quads 5 and 8).
//-----------------------------------------------------------------------------

static void Test_MultiChannel_MergeOrder()
{
    const ImVec4 kClipA(0.0f, 0.0f, 64.0f, 64.0f);
    const ImVec4 kClipB(0.0f, 0.0f, 128.0f, 128.0f);
    const ImVec4 kClipC(0.0f, 0.0f, 256.0f, 256.0f);

    TestDrawList fixture;
    ImDrawList& dl = fixture.DrawList;
    ImDrawListSplitter splitter;
    splitter.Split(&dl, 3);

    static unsigned s_payload_ch1 = 21;
    static unsigned s_payload_ch2 = 22;

    // Emission order is interleaved across channels; each channel has its own clip so a batch
    // that lands in the wrong channel also carries the wrong clip.
    splitter.SetCurrentChannel(&dl, 0);
    dl.PushClipRect(ImVec2(kClipA.x, kClipA.y), ImVec2(kClipA.z, kClipA.w), false);
    fixture.EmitQuad(0);
    fixture.EmitQuad(3);
    fixture.EmitQuad(6);
    dl.PopClipRect();

    splitter.SetCurrentChannel(&dl, 1);
    dl.PushClipRect(ImVec2(kClipB.x, kClipB.y), ImVec2(kClipB.z, kClipB.w), false);
    fixture.EmitQuad(1);
    fixture.EmitQuad(4);
    dl.AddCallback(RecordingCallback, &s_payload_ch1, 0);
    fixture.EmitQuad(7);
    dl.PopClipRect();

    splitter.SetCurrentChannel(&dl, 2);
    dl.PushClipRect(ImVec2(kClipC.x, kClipC.y), ImVec2(kClipC.z, kClipC.w), false);
    fixture.EmitQuad(2);
    fixture.EmitQuad(5);
    dl.AddCallback(RecordingCallback, &s_payload_ch2, 0);
    fixture.EmitQuad(8);
    dl.PopClipRect();

    splitter.Merge(&dl);

    const ImVec4 kClipOf[9] =
    {
        kClipA, kClipB, kClipC,   // ids 0,1,2
        kClipA, kClipB, kClipC,   // ids 3,4,5
        kClipA, kClipB, kClipC,   // ids 6,7,8
    };
    std::vector<QuadSpec> quads(9);
    for (unsigned id = 0; id < 9; id++)
    {
        quads[id].Id = id;
        quads[id].Clip = kClipOf[id];
    }

    // Merge order is strictly by channel: all of channel 0, then channel 1 (with its callback
    // between quad 4 and quad 7), then channel 2 (with its callback between quad 5 and quad 8).
    std::vector<Event> expected;
    expected.push_back(Event::QuadEvent(0));
    expected.push_back(Event::QuadEvent(3));
    expected.push_back(Event::QuadEvent(6));
    expected.push_back(Event::QuadEvent(1));
    expected.push_back(Event::QuadEvent(4));
    expected.push_back(Event::CallbackEvent(s_payload_ch1));
    expected.push_back(Event::QuadEvent(7));
    expected.push_back(Event::QuadEvent(2));
    expected.push_back(Event::QuadEvent(5));
    expected.push_back(Event::CallbackEvent(s_payload_ch2));
    expected.push_back(Event::QuadEvent(8));

    std::vector<Event> actual;
    WalkAndValidate(dl, quads, &actual);
    CheckEventSequence("MultiChannel_MergeOrder", expected, actual);
    CheckCoverage(dl, 9, actual);
}

//-----------------------------------------------------------------------------
// Test 8: clip scopes and 16-bit vertex-segment boundaries interact.
//
// Overlays typically push/pop clips per channel *and* cross the 64K vertex boundary; both events
// affect the command being built. Here a nested clip is pushed exactly when the next quad starts a
// new vertex segment:
//   - quads 0..16382 under clip A (65,532 vertices)
//   - quad 16383 under clip B (nested) -- this quad both changes the clip and crosses 64K
//   - quads 16384.. under clip A again (after PopClipRect)
// Each quad's ClipRect must be the scope it was emitted in, and the batch holding quad 16383 must
// have a non-zero VtxOffset. That command only exists because PushClipRect() split it off, and its
// VtxOffset must be written onto that freshly created (still empty) command.
//-----------------------------------------------------------------------------

static void Test_ClipScopeChangeAtVertexSegmentBoundary()
{
    // After 16383 quads the draw list has 65532 vertices; the next PrimReserve(6, 4) starts a new
    // 16-bit segment because 65532 + 4 >= 65536.
    const unsigned kQuadsBeforeNested = 16383;
    const unsigned kNestedId = kQuadsBeforeNested;
    const unsigned kQuadsAfterNested = 40;
    const ImVec4 kClipA(0.0f, 0.0f, 512.0f, 512.0f);
    const ImVec4 kClipB(4.0f, 4.0f, 256.0f, 256.0f);

    TestDrawList fixture;
    ImDrawList& dl = fixture.DrawList;

    dl.PushClipRect(ImVec2(kClipA.x, kClipA.y), ImVec2(kClipA.z, kClipA.w), false);
    for (unsigned id = 0; id < kQuadsBeforeNested; id++)
        fixture.EmitQuad(id);

    dl.PushClipRect(ImVec2(kClipB.x, kClipB.y), ImVec2(kClipB.z, kClipB.w), false);
    fixture.EmitQuad(kNestedId);
    dl.PopClipRect();

    for (unsigned id = kNestedId + 1; id <= kNestedId + kQuadsAfterNested; id++)
        fixture.EmitQuad(id);

    dl.PopClipRect();

    const unsigned kQuadCount = kNestedId + 1 + kQuadsAfterNested;
    std::vector<QuadSpec> quads(kQuadCount);
    for (unsigned id = 0; id < kQuadCount; id++)
    {
        quads[id].Id = id;
        quads[id].Clip = (id == kNestedId) ? kClipB : kClipA;
    }

    if (sizeof(ImDrawIdx) == 2)
    {
        // The command that draws the nested quad must use a non-zero VtxOffset (it starts the
        // second segment) and must carry clip B.
        int nested_cmd = -1;
        for (int cmd_n = 0; cmd_n < dl.CmdBuffer.Size; cmd_n++)
        {
            const ImDrawCmd& cmd = dl.CmdBuffer[cmd_n];
            if (cmd.UserCallback != NULL || cmd.ElemCount == 0)
                continue;
            const unsigned id = BackendVertex(dl, cmd, BackendLocalIndex(dl, cmd, 0)).col & 0xFFFFFFu;
            if (id == kNestedId)
                nested_cmd = cmd_n;
        }
        T_CHECK(nested_cmd >= 0);
        if (nested_cmd >= 0)
        {
            T_CHECK(dl.CmdBuffer[nested_cmd].VtxOffset > 0u);
            T_CHECK_FLOAT(dl.CmdBuffer[nested_cmd].ClipRect.x, kClipB.x);
            T_CHECK_FLOAT(dl.CmdBuffer[nested_cmd].ClipRect.z, kClipB.z);
            // Local indices restart at 0 in the new segment.
            T_CHECK_UINT(BackendLocalIndex(dl, dl.CmdBuffer[nested_cmd], 0), 0u);
            // The previous command belongs to segment 0 and clip A: it must not have been reused.
            if (nested_cmd > 0)
            {
                const ImDrawCmd& previous = dl.CmdBuffer[nested_cmd - 1];
                if (previous.UserCallback == NULL && previous.ElemCount != 0)
                {
                    T_CHECK_UINT(previous.VtxOffset, 0u);
                    T_CHECK_FLOAT(previous.ClipRect.z, kClipA.z);
                }
            }
        }
    }

    std::vector<Event> expected;
    for (unsigned id = 0; id < kQuadCount; id++)
        expected.push_back(Event::QuadEvent(id));

    std::vector<Event> actual;
    WalkAndValidate(dl, quads, &actual);
    CheckEventSequence("ClipScopeChangeAtBoundary", expected, actual);
    CheckCoverage(dl, kQuadCount, actual);
}

//-----------------------------------------------------------------------------
// Runner
//-----------------------------------------------------------------------------

struct TestCase
{
    const char* Name;
    void (*Fn)();
};

int main()
{
    const TestCase tests[] =
    {
        { "SmallIndex_LargeMesh_VtxOffset",        Test_SmallIndex_LargeMesh_VtxOffset },
        { "Splitter_CallbackOrderAndClips",        Test_Splitter_CallbackOrderAndClips },
        { "Splitter_LargeMesh_CallbackOrder",      Test_Splitter_LargeMesh_CallbackOrder },
        { "ClipChangeAndMergeDoesNotReorder",      Test_ClipChangeAndMergeDoesNotReorder },
        { "NoVtxOffset_Baseline",                  Test_NoVtxOffset_Baseline },
        { "Splitter_ChannelSwitchRespectsVtxOffset", Test_Splitter_ChannelSwitchRespectsVtxOffset },
        { "MultiChannel_MergeOrder",               Test_MultiChannel_MergeOrder },
        { "ClipScopeChangeAtVertexSegmentBoundary", Test_ClipScopeChangeAtVertexSegmentBoundary },
    };

    if (sizeof(ImDrawIdx) != 2)
        printf("[info] ImDrawIdx is not 16-bit in this build; 16-bit segment assertions are skipped.\n");

    int tests_failed = 0;
    for (int n = 0; n < (int)(sizeof(tests) / sizeof(tests[0])); n++)
    {
        const int checks_before = g_checks_run;
        const int failed_before = g_checks_failed;

        g_checks_printed = 0;
        printf("[ RUN  ] %s\n", tests[n].Name);
        tests[n].Fn();

        const int checks = g_checks_run - checks_before;
        const int failed = g_checks_failed - failed_before;
        printf("[ %s ] %s (%d checks, %d failed)\n", failed == 0 ? " OK " : "FAIL", tests[n].Name, checks, failed);
        if (failed != 0)
            tests_failed++;
    }

    printf("----\n%d tests, %d checks, %d failed\n", (int)(sizeof(tests) / sizeof(tests[0])), g_checks_run, g_checks_failed);
    return tests_failed == 0 ? 0 : 1;
}
