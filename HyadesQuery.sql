select
    gs.source_id, gs.ra, gs.ra_error, gs.dec, gs.dec_error, gs.parallax, gs.parallax_error, gs.pmra, gs.pmra_error, gs.pmdec, gs.pmdec_error,
    gs.radial_velocity, gs.radial_velocity_error,
    gs.phot_g_mean_mag, gs.phot_bp_mean_mag, gs.phot_rp_mean_mag,
    ap.mh_gspspec, ap.mh_gspspec_lower, ap.mh_gspspec_upper, ap.fem_gspspec, ap.fem_gspspec_lower, ap.fem_gspspec_upper
from
    gaiadr3.gaia_source as gs
join gaiadr3.astrophysical_parameters as ap using (source_id)
where
    (gs.phot_g_mean_mag is not Null)
    and (gs.phot_bp_mean_mag is not Null)
    and (gs.phot_rp_mean_mag is not Null)
    and (1 = CONTAINS(
        POINT(66.75, 15.87),
        CIRCLE(gs.ra, gs.dec, 300./60.)))
